"""
Algorithm 1: Batched NLI + aggregation for claim verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from src.config import MODELS, VERIFICATION
from src.ingestion.document import LegalChunk


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


@dataclass(frozen=True)
class AggregatedScore:
    """Aggregated verification result for a claim against many chunks."""

    final_score: float
    is_contradicted: bool
    best_chunk_idx: int
    best_chunk: Optional[LegalChunk]
    support_ratio: float
    component_scores: Dict[str, float]


class NLIVerifier:
    """Batched local NLI verification with authority-aware aggregation."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name or MODELS.nli_model
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
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

        self._tokenizer = None
        self._model = None
        self._torch = None

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

        pair_mapping: List[Tuple[int, int]] = []
        pairs: List[Tuple[str, str]] = []
        for claim_idx, claim in enumerate(claims):
            hypothesis = claim if isinstance(claim, str) else claim.text
            if not hypothesis:
                continue
            for chunk_idx, chunk in enumerate(chunks):
                pairs.append((chunk.text, hypothesis))
                pair_mapping.append((claim_idx, chunk_idx))

        if not pairs:
            return [self._empty_result() for _ in claims]

        raw_scores = self._batch_nli_pairs(pairs)
        grouped_scores: Dict[int, List[NLIScore]] = {idx: [] for idx in range(len(claims))}

        for (claim_idx, chunk_idx), scores in zip(pair_mapping, raw_scores):
            grouped_scores[claim_idx].append(
                NLIScore(
                    entailment=scores["entailment"],
                    neutral=scores["neutral"],
                    contradiction=scores["contradiction"],
                    chunk_idx=chunk_idx,
                    chunk=chunks[chunk_idx],
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
        results: List[Dict[str, float]] = []

        for start in range(0, len(pairs), self.batch_size):
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

        return results

    def _load_model(self):
        """Lazy-load tokenizer/model/device selection."""
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return self._tokenizer, self._model, self._torch

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._update_label_map_from_model_config()
        self._model.to(self.device)
        self._model.eval()
        return self._tokenizer, self._model, self._torch

    def _update_label_map_from_model_config(self) -> None:
        """Use the model's label metadata when it provides recognizable NLI labels."""
        if self._model is None:
            return

        id_to_label = getattr(self._model.config, "id2label", {}) or {}
        resolved: dict[str, int] = {}
        for raw_idx, raw_label in id_to_label.items():
            label = str(raw_label).lower()
            if "contrad" in label:
                resolved["contradiction"] = int(raw_idx)
            elif "neutral" in label:
                resolved["neutral"] = int(raw_idx)
            elif "entail" in label:
                resolved["entailment"] = int(raw_idx)

        if {"contradiction", "neutral", "entailment"}.issubset(resolved):
            self.label_map = resolved

    def _aggregate_scores(self, nli_scores: Sequence[NLIScore]) -> AggregatedScore:
        """Aggregate NLI scores across evidence with authority-aware weighting."""
        if not nli_scores:
            return self._empty_result()

        authority_weights = np.array(
            [
                self.authority_weights.get(score.chunk.court_level or "unknown", 0.4)
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
        max_contra = float(np.max(auth_contra))
        is_contradicted = max_contra > self.contradiction_threshold
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
                "support_ratio": support_ratio,
                "auth_weighted_entail": auth_weighted_entail,
                "max_contradiction": max_contra,
                "contra_penalty": contra_penalty,
            },
        )

    @staticmethod
    def _empty_result() -> AggregatedScore:
        return AggregatedScore(
            final_score=0.0,
            is_contradicted=False,
            best_chunk_idx=-1,
            best_chunk=None,
            support_ratio=0.0,
            component_scores={},
        )
