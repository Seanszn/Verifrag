"""
Algorithm 1: Batched NLI + aggregation for claim verification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import logging
import re
import time
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from src.config import MODELS, VERIFICATION
from src.ingestion.document import LegalChunk

logger = logging.getLogger(__name__)

_VERIFICATION_TIER_RANKS = {
    "sentence_evidence": 0,
    "generation_source": 1,
    "prompt_scope": 2,
    "target_doc": 3,
    "target_case": 4,
    "retrieved": 5,
}
_NON_AUTHORITATIVE_CONTRADICTION_POSTURE_RE = re.compile(
    r"\b(?:"
    r"petitioner(?:s)?\s+(?:argue|argued|contend|contended|claim|claimed)"
    r"|respondent(?:s)?\s+(?:argue|argued|contend|contended|claim|claimed)"
    r"|party\s+(?:argue|argued|contend|contended|claim|claimed)"
    r"|government\s+(?:argue|argued|contend|contended|claim|claimed)"
    r"|dissent(?:ing)?\b"
    r"|concurr(?:ing|ence)\b"
    r"|court of appeals\s+(?:held|concluded|reasoned|ruled)"
    r"|district court\s+(?:held|concluded|reasoned|ruled)"
    r"|lower court\s+(?:held|concluded|reasoned|ruled)"
    r"|rejected\s+(?:that|this|the)\s+(?:argument|view|contention|claim)"
    r")",
    re.IGNORECASE,
)
_POSTURE_QUERY_RE = re.compile(
    r"\b(?:argue|argument|petitioner|respondent|dissent|concurr|court of appeals|district court|lower court)\b",
    re.IGNORECASE,
)

class _HasText(Protocol):
    text: str


@dataclass(frozen=True)
class NLIScore:
    """NLI scores for a single (claim, passage) pair."""

    entailment: float
    neutral: float
    contradiction: float
    chunk_idx: int
    chunk: LegalChunk
    hypothesis: str = ""


@dataclass(frozen=True)
class AggregatedScore:
    """Aggregated verification result for a claim against many chunks."""

    final_score: float
    is_contradicted: bool
    best_chunk_idx: int
    best_chunk: Optional[LegalChunk]
    support_ratio: float
    component_scores: Dict[str, float]
    best_contradicting_chunk_idx: int = -1
    best_contradicting_chunk: Optional[LegalChunk] = None


class NLIVerifier:
    """Batched local NLI verification with authority-aware aggregation."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
        dtype: str | None = None,
        unload_after_request: bool | None = None,
    ) -> None:
        self.model_name = model_name or MODELS.nli_model
        self.device = device or MODELS.nli_device
        self.batch_size = max(1, int(batch_size or MODELS.nli_batch_size))
        self.max_length = max(32, int(max_length or MODELS.nli_max_length))
        self.dtype = (dtype or MODELS.nli_dtype).lower().strip()
        self.unload_after_request = (
            MODELS.nli_unload_after_request
            if unload_after_request is None
            else unload_after_request
        )
        self.label_map = {
            label: idx for idx, label in enumerate(MODELS.nli_labels)
        }

        self.alpha = VERIFICATION.agg_alpha
        self.beta = VERIFICATION.agg_beta
        self.gamma = VERIFICATION.agg_gamma
        self.delta = VERIFICATION.agg_delta
        self.support_threshold = VERIFICATION.support_threshold
        self.contradiction_threshold = VERIFICATION.contradiction_threshold
        self.authority_weights = VERIFICATION.authority_weights
        
        # Rhetorical contradiction detection config
        self.rhetorical_contradiction_entailment_threshold = getattr(
            VERIFICATION, 'rhetorical_contradiction_entailment_threshold', 0.95
        )
        self.rhetorical_contradiction_tier_gap = getattr(
            VERIFICATION, 'rhetorical_contradiction_tier_gap', 2
        )
        self.rhetorical_contradiction_penalty_discount = getattr(
            VERIFICATION, 'rhetorical_contradiction_penalty_discount', 0.5
        )

        self._tokenizer = None
        self._model = None
        self._torch = None
        self.local_files_only = MODELS.huggingface_local_files_only

    def verify_claim(
        self,
        claim: _HasText | str,
        chunks: Sequence[LegalChunk],
    ) -> AggregatedScore:
        """Verify one claim against retrieved evidence chunks."""
        hypothesis = claim if isinstance(claim, str) else claim.text
        if not hypothesis or not chunks:
            return self._empty_result()

        nli_scores = self._batch_nli(hypothesis, chunks)
        return self._aggregate_scores(nli_scores)

    def verify_claims_batch(
        self,
        claims: Sequence[_HasText | str],
        chunks: Sequence[LegalChunk],
    ) -> List[AggregatedScore]:
        """Verify many claims against a shared chunk set in one inference batch."""
        if not claims:
            return []
        if not chunks:
            return [self._empty_result() for _ in claims]

        pair_mapping: List[Tuple[int, int, str]] = []
        pairs: List[Tuple[str, str]] = []
        for claim_idx, claim in enumerate(claims):
            hypothesis = claim if isinstance(claim, str) else claim.text
            if not hypothesis:
                continue
            for chunk_idx, chunk in enumerate(chunks):
                pairs.append((chunk.text, hypothesis))
                pair_mapping.append((claim_idx, chunk_idx, hypothesis))

        if not pairs:
            return [self._empty_result() for _ in claims]

        raw_scores = self._batch_nli_pairs(pairs)
        grouped_scores: Dict[int, List[NLIScore]] = {idx: [] for idx in range(len(claims))}

        for (claim_idx, chunk_idx, hypothesis), scores in zip(pair_mapping, raw_scores):
            grouped_scores[claim_idx].append(
                NLIScore(
                    entailment=scores["entailment"],
                    neutral=scores["neutral"],
                    contradiction=scores["contradiction"],
                    chunk_idx=chunk_idx,
                    chunk=chunks[chunk_idx],
                    hypothesis=hypothesis,
                )
            )

        return [
            self._aggregate_scores(grouped_scores[idx]) if grouped_scores[idx] else self._empty_result()
            for idx in range(len(claims))
        ]

    def _batch_nli(
        self,
        hypothesis: str,
        chunks: Sequence[LegalChunk],
    ) -> List[NLIScore]:
        """Run NLI for one hypothesis against multiple premises."""
        if not hypothesis or not chunks:
            return []

        pairs = [(chunk.text, hypothesis) for chunk in chunks]
        raw_scores = self._batch_nli_pairs(pairs)

        return [
            NLIScore(
                entailment=scores["entailment"],
                neutral=scores["neutral"],
                contradiction=scores["contradiction"],
                chunk_idx=idx,
                chunk=chunks[idx],
                hypothesis=hypothesis,
            )
            for idx, scores in enumerate(raw_scores)
        ]

    def _batch_nli_pairs(
        self,
        pairs: Sequence[Tuple[str, str]],
    ) -> List[Dict[str, float]]:
        """Run batched NLI inference on (premise, hypothesis) pairs."""
        if not pairs:
            return []
        return self._predict_pairs(pairs)

    def _predict_pairs(
        self,
        pairs: Sequence[Tuple[str, str]],
    ) -> List[Dict[str, float]]:
        """
        Score (premise, hypothesis) pairs with the configured NLI model.

        This method is kept isolated so tests can override it without
        downloading the model.
        """
        tokenizer, model, torch = self._load_model()
        try:
            results: List[Dict[str, float]] = []
            total_batches = max(1, math.ceil(len(pairs) / self.batch_size))
            started = time.perf_counter()
            logger.info(
                "verification.inference_start model=%s device=%s dtype=%s pairs=%s batches=%s batch_size=%s max_length=%s unload_after_request=%s",
                self.model_name,
                self.device,
                self.dtype,
                len(pairs),
                total_batches,
                self.batch_size,
                self.max_length,
                self.unload_after_request,
            )

            for batch_index, start in enumerate(range(0, len(pairs), self.batch_size), start=1):
                batch_started = time.perf_counter()
                batch = pairs[start : start + self.batch_size]
                premises = [premise for premise, _ in batch]
                hypotheses = [hypothesis for _, hypothesis in batch]

                encoded = tokenizer(
                    premises,
                    hypotheses,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}

                with torch.no_grad():
                    logits = model(**encoded).logits
                    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

                for row in probs:
                    results.append(
                        {
                            "contradiction": float(row[self.label_map["contradiction"]]),
                            "neutral": float(row[self.label_map["neutral"]]),
                            "entailment": float(row[self.label_map["entailment"]]),
                        }
                    )
                logger.info(
                    "verification.inference_batch_complete model=%s batch=%s/%s batch_pairs=%s elapsed_ms=%.1f total_results=%s",
                    self.model_name,
                    batch_index,
                    total_batches,
                    len(batch),
                    (time.perf_counter() - batch_started) * 1000,
                    len(results),
                )

            logger.info(
                "verification.inference_complete model=%s device=%s pairs=%s elapsed_ms=%.1f",
                self.model_name,
                self.device,
                len(pairs),
                (time.perf_counter() - started) * 1000,
            )

            return results
        finally:
            if self.unload_after_request:
                self.unload_model()

    def _load_model(self):
        """Lazy-load tokenizer/model/device selection."""
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return self._tokenizer, self._model, self._torch

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.device.lower().strip()
            if self.device == "cuda" and not torch.cuda.is_available():
                self.device = "cpu"
        torch_dtype = self._resolve_torch_dtype(torch)

        started = time.perf_counter()
        logger.info(
            "verification.model_load_start model=%s device=%s dtype=%s local_files_only=%s",
            self.model_name,
            self.device,
            self.dtype,
            self.local_files_only,
        )
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
        )
        model_kwargs = {"local_files_only": self.local_files_only}
        if torch_dtype is not None:
            model_kwargs["dtype"] = torch_dtype
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        self.label_map = self._resolve_label_map(self._model)
        self._model.to(self.device)
        self._model.eval()
        logger.info(
            "verification.model_load_complete model=%s device=%s dtype=%s local_files_only=%s elapsed_ms=%.1f",
            self.model_name,
            self.device,
            self.dtype,
            self.local_files_only,
            (time.perf_counter() - started) * 1000,
        )
        return self._tokenizer, self._model, self._torch

    def unload_model(self) -> None:
        """Release the NLI model so Ollama can reclaim GPU memory."""
        torch = self._torch
        device = self.device
        self._tokenizer = None
        self._model = None
        self._torch = None
        if torch is not None and device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(
            "verification.model_unloaded model=%s device=%s",
            self.model_name,
            device,
        )

    def _resolve_torch_dtype(self, torch):
        if self.device != "cuda":
            self.dtype = "float32"
            return None
        if self.dtype in {"", "auto"}:
            self.dtype = "float16"
            return torch.float16
        if self.dtype in {"float16", "fp16", "half"}:
            self.dtype = "float16"
            return torch.float16
        if self.dtype in {"bfloat16", "bf16"}:
            self.dtype = "bfloat16"
            return torch.bfloat16
        if self.dtype in {"float32", "fp32", "full"}:
            self.dtype = "float32"
            return torch.float32
        logger.warning(
            "verification.invalid_dtype dtype=%s fallback=float16",
            self.dtype,
        )
        self.dtype = "float16"
        return torch.float16

    @staticmethod
    def _resolve_label_map(model) -> Dict[str, int]:
        config = getattr(model, "config", None)
        id2label = getattr(config, "id2label", None)
        if not isinstance(id2label, dict):
            return {
                label: idx for idx, label in enumerate(MODELS.nli_labels)
            }

        normalized: Dict[str, int] = {}
        for raw_idx, raw_label in id2label.items():
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError):
                continue
            label = str(raw_label).strip().lower()
            if "contrad" in label:
                normalized["contradiction"] = idx
            elif "neutral" in label:
                normalized["neutral"] = idx
            elif "entail" in label:
                normalized["entailment"] = idx

        if set(normalized) == {"contradiction", "neutral", "entailment"}:
            return normalized

        return {
            label: idx for idx, label in enumerate(MODELS.nli_labels)
        }

    def _aggregate_scores(self, nli_scores: Sequence[NLIScore]) -> AggregatedScore:
        """Aggregate NLI scores across evidence with authority-aware weighting."""
        if not nli_scores:
            return self._empty_result()

        authority_weights = np.array(
            [
                self._authority_weight_for_chunk(score.chunk)
                for score in nli_scores
            ],
            dtype=np.float32,
        )
        entailments = np.array([score.entailment for score in nli_scores], dtype=np.float32)
        contradictions = np.array([score.contradiction for score in nli_scores], dtype=np.float32)

        auth_entail = entailments * authority_weights
        best_idx = int(np.argmax(auth_entail))
        max_pool = float(auth_entail[best_idx])

        support_ratio = float(np.mean(entailments > self.support_threshold))

        rank_weights = np.array(
            [1.0 / (score.chunk_idx + 1.0) for score in nli_scores],
            dtype=np.float32,
        )
        combined_weights = authority_weights * rank_weights
        weight_sum = float(np.sum(combined_weights)) or 1.0
        auth_weighted_entail = float(np.sum(entailments * combined_weights) / weight_sum)

        auth_contra = contradictions * authority_weights
        best_contra_idx = int(np.argmax(auth_contra))
        max_contra = float(np.max(auth_contra))
        best_entailment = float(entailments[best_idx])
        best_contradiction = float(contradictions[best_contra_idx])
        contradiction_margin = max_contra - max_pool
        
        # NEW: Detect "same chunk confusion" - when best support AND best contradiction
        # come from the same chunk, the NLI model is uncertain. Apply stricter rules.
        same_chunk_confusion = best_idx == best_contra_idx
        if same_chunk_confusion:
            # When same chunk produces both high entailment and high contradiction,
            # the model is confused. Require larger margin for valid contradiction.
            # This prevents "possibly_supported" from conflicting signals on same evidence.
            if best_entailment >= 0.70 and best_contradiction >= 0.60:
                # Both scores are high - model is truly uncertain
                # Require contradiction to be significantly higher than entailment
                if contradiction_margin < 0.20:
                    # Not enough margin - ignore the contradiction signal from this chunk
                    # The entailment signal is more trustworthy when model is uncertain
                    max_contra = 0.0  # Neutralize the contradiction
                    best_contradiction = 0.0
                    contradiction_margin = -max_pool  # Negative means entailment dominates
        
        support_tier_rank = _verification_tier_rank(nli_scores[best_idx].chunk)
        contradiction_tier_rank = _verification_tier_rank(nli_scores[best_contra_idx].chunk)
        contradiction_posture_valid = _contradiction_posture_valid(
            nli_scores[best_contra_idx].chunk,
            nli_scores[best_contra_idx].hypothesis,
        )
        contradiction_overlap_valid = _contradiction_overlap_valid(
            nli_scores[best_contra_idx].chunk,
            nli_scores[best_contra_idx].hypothesis,
        )
        tier_allows_contradiction = _tier_allows_contradiction_override(
            best_entailment=best_entailment,
            contradiction_margin=contradiction_margin,
            support_tier_rank=support_tier_rank,
            contradiction_tier_rank=contradiction_tier_rank,
            contradiction_posture_valid=contradiction_posture_valid,
        )
        # Enhanced support-dominates logic with tier-aware rhetorical contradiction detection
        # When we have near-perfect entailment from highest tier, contradiction from lower
        # tiers (concurrences, dissents) is likely rhetorical tension, not factual falsity
        near_perfect_entailment = (
            best_entailment >= self.rhetorical_contradiction_entailment_threshold 
            and support_tier_rank <= 2
        )
        
        support_dominates_contradiction = (
            (best_entailment >= 0.90
             and (support_ratio >= 0.35 or auth_weighted_entail >= 0.45)
             and contradiction_margin < 0.10)
            or (near_perfect_entailment and contradiction_tier_rank > support_tier_rank)
        )
        
        # NEW: Rhetorical contradiction detection
        # When contradiction comes from lower tier than support, and entailment is very high,
        # the contradiction is likely from dissenting views, not factual falsity
        tier_gap = contradiction_tier_rank - support_tier_rank
        is_rhetorical_contradiction = (
            near_perfect_entailment
            and tier_gap >= self.rhetorical_contradiction_tier_gap
            and best_contradiction < 0.999  # Not absolute certainty of contradiction
        )
        is_contradicted = (
            max_contra > self.contradiction_threshold
            and not support_dominates_contradiction
            and tier_allows_contradiction
            and contradiction_overlap_valid
        )
        
        # NEW: Apply rhetorical contradiction discount
        # When high entailment from top-tier source coexists with contradiction from lower tiers,
        # it's often dissent/concurrence rhetoric, not factual falsity. Reduce penalty.
        if is_rhetorical_contradiction and is_contradicted:
            # Reduce penalty for rhetorical contradictions (e.g., dissenting views)
            discount = self.rhetorical_contradiction_penalty_discount
            contra_penalty = max_contra * (1.0 - discount)
        else:
            contra_penalty = max_contra if is_contradicted else 0.0

        raw_final = (
            self.alpha * max_pool
            + self.beta * support_ratio
            + self.gamma * auth_weighted_entail
            - self.delta * contra_penalty
        )
        final_score = float(np.clip(raw_final, 0.0, 1.0))

        return AggregatedScore(
            final_score=final_score,
            is_contradicted=is_contradicted,
            best_chunk_idx=nli_scores[best_idx].chunk_idx,
            best_chunk=nli_scores[best_idx].chunk,
            support_ratio=support_ratio,
            component_scores={
                "max_pool": max_pool,
                "best_entailment": best_entailment,
                "support_ratio": support_ratio,
                "auth_weighted_entail": auth_weighted_entail,
                "max_contradiction": max_contra,
                "best_contradiction": best_contradiction,
                "contradiction_margin": contradiction_margin,
                "best_support_tier_rank": float(support_tier_rank),
                "best_contradiction_tier_rank": float(contradiction_tier_rank),
                "contradiction_posture_valid": float(contradiction_posture_valid),
                "contradiction_overlap_valid": float(contradiction_overlap_valid),
                "tier_allows_contradiction": float(tier_allows_contradiction),
                "contra_penalty": contra_penalty,
                "support_dominates_contradiction": float(support_dominates_contradiction),
                "is_rhetorical_contradiction": float(is_rhetorical_contradiction),
                "near_perfect_entailment": float(near_perfect_entailment),
            },
            best_contradicting_chunk_idx=nli_scores[best_contra_idx].chunk_idx,
            best_contradicting_chunk=nli_scores[best_contra_idx].chunk,
        )

    def _authority_weight_for_chunk(self, chunk: LegalChunk) -> float:
        """Weight public legal authority separately from uploaded record facts."""
        if getattr(chunk, "doc_type", None) == "user_upload":
            return 1.0
        return float(self.authority_weights.get(chunk.court_level or "unknown", 0.4))

    @staticmethod
    def _empty_result() -> AggregatedScore:
        return AggregatedScore(
            final_score=0.0,
            is_contradicted=False,
            best_chunk_idx=-1,
            best_chunk=None,
            support_ratio=0.0,
            component_scores={},
            best_contradicting_chunk_idx=-1,
            best_contradicting_chunk=None,
        )


def _verification_tier_rank(chunk: LegalChunk) -> int:
    raw_rank = getattr(chunk, "verification_tier_rank", None)
    if isinstance(raw_rank, int):
        return raw_rank
    tier = str(getattr(chunk, "verification_tier", "") or "").strip().lower()
    return _VERIFICATION_TIER_RANKS.get(tier, _VERIFICATION_TIER_RANKS["retrieved"])


def _contradiction_posture_valid(chunk: LegalChunk, hypothesis: str) -> bool:
    text = str(getattr(chunk, "text", "") or "")
    if not _NON_AUTHORITATIVE_CONTRADICTION_POSTURE_RE.search(text):
        return True
    return bool(_POSTURE_QUERY_RE.search(str(hypothesis or "")))


_CONTRADICTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "by",
    "case",
    "court",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "v",
    "was",
    "were",
    "when",
    "which",
    "with",
}


def _contradiction_overlap_valid(chunk: LegalChunk, hypothesis: str) -> bool:
    chunk_text = getattr(chunk, "text", "")
    if _direct_legal_polarity_conflict(chunk_text, hypothesis):
        return True
    hypothesis_anchors = _contradiction_anchor_tokens(hypothesis)
    if hypothesis_anchors:
        chunk_anchors = _contradiction_anchor_tokens(chunk_text)
        if not (hypothesis_anchors & chunk_anchors):
            return False
    hypothesis_tokens = _contradiction_content_tokens(hypothesis)
    if len(hypothesis_tokens) < 4:
        return False
    chunk_tokens = _contradiction_content_tokens(chunk_text)
    if not chunk_tokens:
        return False
    overlap = hypothesis_tokens & chunk_tokens
    overlap_ratio = len(overlap) / max(len(hypothesis_tokens), 1)
    return len(overlap) >= 3 and overlap_ratio >= 0.30


def _direct_legal_polarity_conflict(premise: str, hypothesis: str) -> bool:
    premise_tokens = _contradiction_content_tokens(premise)
    hypothesis_tokens = _contradiction_content_tokens(hypothesis)
    shared_subject = premise_tokens & hypothesis_tokens
    if not shared_subject:
        return False
    polarity_pairs = (
        ({"grant", "granted", "grants"}, {"deny", "denied", "denies"}),
        ({"affirm", "affirmed", "affirms"}, {"reverse", "reversed", "reverses"}),
        ({"constitutional", "valid"}, {"unconstitutional", "invalid"}),
        ({"authorize", "authorized", "allows", "allow"}, {"prohibit", "prohibited", "bars", "bar"}),
    )
    for positive, negative in polarity_pairs:
        premise_positive = bool(premise_tokens & positive)
        premise_negative = bool(premise_tokens & negative)
        hypothesis_positive = bool(hypothesis_tokens & positive)
        hypothesis_negative = bool(hypothesis_tokens & negative)
        if premise_positive and hypothesis_negative:
            return True
        if premise_negative and hypothesis_positive:
            return True
    return False


def _contradiction_content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.sub(r"[^A-Za-z0-9]+", " ", str(text or "").lower()).split()
        if len(token) > 2 and token not in _CONTRADICTION_STOPWORDS
    }


def _contradiction_anchor_tokens(text: str) -> set[str]:
    raw = str(text or "")
    lowered = raw.lower()
    anchors: set[str] = set()
    for match in re.finditer(r"\b\d+\s+u\.?\s*s\.?\s*c\.?\s*§?\s*\d+[a-z0-9()\-]*", lowered):
        anchors.add(re.sub(r"[^a-z0-9]+", "", match.group(0)))
    for match in re.finditer(r"§+\s*\d+[a-z0-9()\-]*", lowered):
        anchors.add(re.sub(r"[^a-z0-9]+", "", match.group(0)))

    phrase_anchors = (
        ("removal petition", "removalpetition"),
        ("statement of jurisdiction", "statementofjurisdiction"),
        ("jurisdictional statement", "jurisdictionalstatement"),
        ("domestic violence restraining order", "domesticviolencerestrainingorder"),
        ("preliminary injunction", "preliminaryinjunction"),
        ("article iii standing", "articleiiistanding"),
        ("equal protection clause", "equalprotectionclause"),
        ("fourteenth amendment", "fourteenthamendment"),
        ("second amendment", "secondamendment"),
        ("first amendment", "firstamendment"),
        ("takings clause", "takingsclause"),
    )
    for phrase, token in phrase_anchors:
        if phrase in lowered:
            anchors.add(token)
    return anchors


def _tier_allows_contradiction_override(
    *,
    best_entailment: float,
    contradiction_margin: float,
    support_tier_rank: int,
    contradiction_tier_rank: int,
    contradiction_posture_valid: bool,
) -> bool:
    if contradiction_tier_rank <= support_tier_rank:
        if not contradiction_posture_valid:
            return False
        if support_tier_rank <= _VERIFICATION_TIER_RANKS["generation_source"] and best_entailment >= 0.85:
            return contradiction_margin >= 0.15
        return True
    if not contradiction_posture_valid:
        return False
    if support_tier_rank <= _VERIFICATION_TIER_RANKS["prompt_scope"] and best_entailment >= 0.85:
        return contradiction_margin >= 0.20
    return contradiction_margin >= 0.10
