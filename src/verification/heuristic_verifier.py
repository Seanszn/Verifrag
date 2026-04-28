"""Deterministic offline verifier used as a runtime fallback."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from src.verification.claim_decomposer import split_sentences
from src.verification.nli_verifier import NLIVerifier


_WORD_RE = re.compile(r"[A-Za-z0-9]+")
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


class HeuristicNLIVerifier(NLIVerifier):
    """Rule-based verifier for offline and low-resource fallback use."""

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
    return sum(1 for index, token in enumerate(tokens) if _is_semantic_negation(tokens, index)) % 2


def _is_semantic_negation(tokens, index: int) -> bool:
    token = tokens[index]
    if token not in _NEGATIONS:
        return False
    # Legal captions use "No." before docket numbers. Treating that as negation
    # creates false contradictions for certiorari/order captions.
    if token == "no" and index + 1 < len(tokens) and str(tokens[index + 1]).isdigit():
        return False
    return True


def _has_explicit_negation_wrapper(text: str) -> bool:
    return text.startswith("it is not true that ")


def _clean_sentence(text: str) -> str:
    cleaned = " ".join(str(text).split()).strip()
    if not cleaned:
        return ""
    if cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."
    return cleaned


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
