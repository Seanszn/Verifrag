"""Shared helpers for corpus-backed NLI smoke and pipeline testing."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import INDEX_DIR
from src.indexing.bm25_index import BM25Index
from src.indexing.index_discovery import discover_index_artifacts
from src.ingestion.document import LegalChunk
from src.verification.claim_decomposer import decompose_document, split_sentences
from src.verification.nli_verifier import NLIVerifier


DEFAULT_QUERY = "Did the Supreme Court uphold the TikTok law against a First Amendment challenge?"
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_NOISE_MARKERS = (
    "preliminary print",
    "reporter of decisions",
    "notice:",
    "page proof pending publication",
    "cite as:",
)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
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
    "under",
    "was",
    "were",
    "what",
    "with",
}
_NEGATIONS = {"no", "not", "never", "none", "cannot", "cant", "won't", "wont", "without"}
_CONTINUATION_STARTS = (
    "and ",
    "or ",
    "but ",
    "because ",
    "if ",
    "when ",
    "while ",
    "then ",
    "than ",
    "so ",
)


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    score: float
    chunk: LegalChunk

    def to_dict(self) -> dict[str, Any]:
        payload = self.chunk.to_dict()
        payload["rank"] = self.rank
        payload["score"] = self.score
        payload["text_preview"] = self.chunk.text[:320]
        return payload


@dataclass(frozen=True)
class GeneratedTestCase:
    query: str
    response: str
    retrieved_chunks: tuple[RetrievedChunk, ...]
    supporting_sentences: tuple[str, ...]

    @property
    def claims(self):
        return decompose_document({"id": "generated_response", "full_text": self.response})

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "response": self.response,
            "supporting_sentences": list(self.supporting_sentences),
            "claims": [claim.to_dict() for claim in self.claims],
            "retrieved_chunks": [hit.to_dict() for hit in self.retrieved_chunks],
        }


class BM25Retriever:
    """Small retriever adapter backed by a persisted BM25 artifact."""

    def __init__(self, *, index_path: Path | None = None, top_k: int = 8) -> None:
        self.index_path = Path(index_path or default_bm25_index_path())
        self.top_k = top_k
        self.index = BM25Index(index_path=self.index_path)
        self.index.load()

    def retrieve(self, query: str, k: int = 10) -> list[LegalChunk]:
        limit = min(k or self.top_k, self.top_k)
        hits = self.index.search(query, k=limit)
        return [_chunk_from_payload(metadata) for _, _, metadata in hits]


class StaticResponseLLM:
    """Simple stand-in for the LLM backend during pipeline tests."""

    def __init__(self, response: str) -> None:
        self.response = response

    def generate_legal_answer(self, query: str) -> str:
        _ = query
        return self.response


class HeuristicNLIVerifier(NLIVerifier):
    """Deterministic verifier used for offline smoke and pipeline scripts."""

    def __init__(self) -> None:
        super().__init__(device="cpu")

    def _predict_pairs(self, pairs):
        return [self._score_pair(premise, hypothesis) for premise, hypothesis in pairs]

    def _score_pair(self, premise: str, hypothesis: str) -> dict[str, float]:
        premise = _best_matching_premise_sentence(premise, hypothesis)
        premise_norm = _compact_text(premise)
        hypothesis_norm = _compact_text(hypothesis)
        premise_tokens = _content_tokens(premise_norm)
        hypothesis_tokens = _content_tokens(hypothesis_norm)

        if not premise_tokens or not hypothesis_tokens:
            return {"entailment": 0.05, "neutral": 0.90, "contradiction": 0.05}

        overlap = len(premise_tokens & hypothesis_tokens)
        hypothesis_coverage = overlap / max(len(hypothesis_tokens), 1)
        sequence = SequenceMatcher(None, premise_norm, hypothesis_norm).ratio()
        similarity_signal = max(hypothesis_coverage, sequence)

        if hypothesis_norm in premise_norm or premise_norm in hypothesis_norm:
            return {"entailment": 0.96, "neutral": 0.03, "contradiction": 0.01}

        has_negation_mismatch = _has_negation_mismatch(premise_norm, hypothesis_norm)
        explicit_negation = _has_explicit_negation_wrapper(hypothesis_norm)
        if has_negation_mismatch and (
            similarity_signal >= 0.72
            or (explicit_negation and similarity_signal >= 0.58)
        ):
            contradiction = min(0.75 + (0.20 * hypothesis_coverage), 0.94)
            entailment = 0.05
            neutral = max(0.01, 1.0 - contradiction - entailment)
            return {
                "entailment": round(entailment, 4),
                "neutral": round(neutral, 4),
                "contradiction": round(contradiction, 4),
            }

        if hypothesis_coverage < 0.18 and sequence < 0.45:
            return {"entailment": 0.08, "neutral": 0.87, "contradiction": 0.05}

        entailment = min(0.28 + 0.54 * similarity_signal, 0.93)
        contradiction = 0.03 if hypothesis_coverage >= 0.45 else 0.05
        neutral = max(0.02, 1.0 - entailment - contradiction)
        return {
            "entailment": round(entailment, 4),
            "neutral": round(neutral, 4),
            "contradiction": round(contradiction, 4),
        }


def default_bm25_index_path() -> Path:
    artifacts = discover_index_artifacts(index_dir=INDEX_DIR)
    return artifacts.bm25_path


def load_bm25_index(index_path: Path | None = None) -> BM25Index:
    path = Path(index_path or default_bm25_index_path())
    index = BM25Index(index_path=path)
    index.load()
    return index


def retrieve_bm25_hits(
    query: str,
    *,
    index_path: Path | None = None,
    top_k: int = 8,
) -> list[RetrievedChunk]:
    index = load_bm25_index(index_path)
    raw_hits = index.search(query, k=top_k)
    hits: list[RetrievedChunk] = []
    for rank, (_, score, metadata) in enumerate(raw_hits, start=1):
        hits.append(RetrievedChunk(rank=rank, score=float(score), chunk=_chunk_from_payload(metadata)))
    return hits


def build_corpus_backed_test_case(
    query: str = DEFAULT_QUERY,
    *,
    index_path: Path | None = None,
    top_k: int = 8,
    max_sentences: int = 3,
    response_text: str | None = None,
) -> GeneratedTestCase:
    hits = retrieve_bm25_hits(query, index_path=index_path, top_k=top_k)
    if not hits:
        raise RuntimeError(f"No BM25 hits found for query: {query!r}")

    if response_text is None:
        response_text, supporting_sentences = build_reasonable_response(query, hits, max_sentences=max_sentences)
    else:
        supporting_sentences = tuple()

    return GeneratedTestCase(
        query=query,
        response=response_text,
        retrieved_chunks=tuple(hits),
        supporting_sentences=tuple(supporting_sentences),
    )


def build_reasonable_response(
    query: str,
    hits: Sequence[RetrievedChunk],
    *,
    max_sentences: int = 3,
) -> tuple[str, tuple[str, ...]]:
    query_tokens = _content_tokens(query)
    candidates: list[tuple[float, int, str]] = []

    for hit in hits:
        for sentence in split_sentences(hit.chunk.text):
            sentence = _clean_sentence(sentence)
            if not _is_usable_sentence(sentence):
                continue

            sentence_tokens = _content_tokens(sentence)
            if not sentence_tokens:
                continue

            overlap = len(query_tokens & sentence_tokens)
            coverage = overlap / max(len(query_tokens), 1)
            score = (coverage * 10.0) + max(0.0, 3.0 - (hit.rank * 0.35))
            lowered = sentence.lower()
            if lowered.startswith("held:"):
                score += 3.0
            if "the court" in lowered:
                score += 1.5
            if "first amendment" in lowered or "tiktok" in lowered:
                score += 1.2
            if "content neutral" in lowered or "important government interest" in lowered:
                score += 1.0

            candidates.append((score, hit.rank, sentence))

    if not candidates:
        fallback = _clean_sentence(hits[0].chunk.text[:300])
        return fallback, (fallback,)

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    chosen: list[str] = []
    seen_keys: set[str] = set()

    for _, _, sentence in candidates:
        key = _compact_text(sentence)
        if key in seen_keys:
            continue
        if any(_sentence_similarity(sentence, existing) >= 0.84 for existing in chosen):
            continue
        chosen.append(_normalize_response_sentence(sentence))
        seen_keys.add(key)
        if len(chosen) >= max_sentences:
            break

    response = " ".join(chosen).strip()
    return response, tuple(chosen)


def build_mixed_response_with_contradiction(response: str) -> tuple[str, str]:
    sentences = [_clean_sentence(sentence) for sentence in split_sentences(response)]
    if not sentences:
        raise RuntimeError("Cannot inject a contradiction into an empty response.")

    contradiction = invert_sentence_polarity(sentences[0])
    mixed_sentences = list(sentences)
    mixed_sentences.append(contradiction)
    return " ".join(sentence for sentence in mixed_sentences if sentence), contradiction


def invert_sentence_polarity(sentence: str) -> str:
    rewrites = (
        (" do not ", " do "),
        (" does not ", " does "),
        (" did not ", " did "),
        (" cannot ", " can "),
        (" can not ", " can "),
        (" are ", " are not "),
        (" is ", " is not "),
        (" was ", " was not "),
        (" were ", " were not "),
        (" has ", " has not "),
        (" have ", " have not "),
    )

    padded = f" {sentence.strip()} "
    lowered = padded.lower()
    for before, after in rewrites:
        idx = lowered.find(before)
        if idx >= 0:
            replacement = after
            mutated = padded[:idx] + replacement + padded[idx + len(before) :]
            return _clean_sentence(mutated)

    if sentence.endswith("."):
        sentence = sentence[:-1]
    return f"It is not true that {sentence.lower()}."


def claims_to_payload(claims: Iterable[Any]) -> list[dict[str, Any]]:
    return [claim.to_dict() for claim in claims]


def dump_json(payload: Mapping[str, Any], *, output_path: Path | None = None) -> str:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    return text


def _chunk_from_payload(payload: Mapping[str, Any]) -> LegalChunk:
    raw_date = payload.get("date_decided")
    parsed_date = date.fromisoformat(str(raw_date)) if raw_date else None
    return LegalChunk(
        id=str(payload["id"]),
        doc_id=str(payload["doc_id"]),
        text=str(payload["text"]),
        chunk_index=int(payload["chunk_index"]),
        doc_type=str(payload["doc_type"]),
        case_name=payload.get("case_name"),
        court=payload.get("court"),
        court_level=payload.get("court_level"),
        citation=payload.get("citation"),
        date_decided=parsed_date,
        title=payload.get("title"),
        section=payload.get("section"),
        source_file=payload.get("source_file"),
    )


def _clean_sentence(text: str) -> str:
    cleaned = " ".join(str(text).split()).strip()
    if not cleaned:
        return ""
    if cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."
    return cleaned


def _is_usable_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    if len(sentence) < 35:
        return False
    if any(marker in lowered for marker in _NOISE_MARKERS):
        return False
    if lowered.startswith(_CONTINUATION_STARTS):
        return False
    if sentence[0].islower():
        return False

    alpha_chars = sum(1 for char in sentence if char.isalpha())
    if alpha_chars < 20:
        return False
    uppercase_ratio = sum(1 for char in sentence if char.isupper()) / max(alpha_chars, 1)
    if uppercase_ratio > 0.55:
        return False
    return True


def _normalize_response_sentence(sentence: str) -> str:
    sentence = re.sub(r"^\([A-Za-z0-9]+\)\s+", "", sentence).strip()
    sentence = re.sub(r"^Pp\.\s*\d+(?:[–-]\d+)?\.\s*", "", sentence).strip()
    lowered = sentence.lower()
    if lowered.startswith("held:"):
        tail = sentence.split(":", 1)[1].strip()
        if not tail:
            return sentence
        tail = tail[0].lower() + tail[1:] if len(tail) > 1 else tail.lower()
        return f"The Court held that {tail}"
    return sentence


def _compact_text(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def _content_tokens(text: str) -> set[str]:
    tokens = {token for token in _WORD_RE.findall(text.lower()) if len(token) > 2}
    return {token for token in tokens if token not in _STOPWORDS}


def _has_negation_mismatch(left: str, right: str) -> bool:
    left_tokens = _WORD_RE.findall(left)
    right_tokens = _WORD_RE.findall(right)
    if _has_explicit_negation_wrapper(right):
        return True
    return _negation_parity(left_tokens) != _negation_parity(right_tokens)


def _negation_parity(tokens) -> int:
    return sum(1 for token in tokens if token in _NEGATIONS) % 2


def _has_explicit_negation_wrapper(text: str) -> bool:
    return text.startswith("it is not true that ")


def _sentence_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, _compact_text(left), _compact_text(right)).ratio()


def _best_matching_premise_sentence(premise: str, hypothesis: str) -> str:
    best_sentence = _clean_sentence(premise)
    best_score = -1.0
    hypothesis_norm = _compact_text(hypothesis)
    hypothesis_tokens = _content_tokens(hypothesis_norm)

    for sentence in split_sentences(premise):
        cleaned = _clean_sentence(sentence)
        if not cleaned:
            continue
        sentence_norm = _compact_text(cleaned)
        sentence_tokens = _content_tokens(sentence_norm)
        overlap = len(sentence_tokens & hypothesis_tokens) / max(len(hypothesis_tokens), 1)
        sequence = SequenceMatcher(None, sentence_norm, hypothesis_norm).ratio()
        score = max(overlap, sequence)
        if score > best_score:
            best_sentence = cleaned
            best_score = score

    return best_sentence
