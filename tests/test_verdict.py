"""Tests for calibrated verification verdict labels."""

from __future__ import annotations

from src.ingestion.document import LegalChunk
from src.verification.nli_verifier import AggregatedScore
from src.verification.verdict import classify_verification


def _score(value: float, *, contradicted: bool = False, has_chunk: bool = True) -> AggregatedScore:
    chunk = (
        LegalChunk(
            id="chunk",
            doc_id="doc",
            text="Evidence text.",
            chunk_index=0,
            doc_type="case",
            court_level="scotus",
        )
        if has_chunk
        else None
    )
    return AggregatedScore(
        final_score=value,
        is_contradicted=contradicted,
        best_chunk_idx=0 if has_chunk else -1,
        best_chunk=chunk,
        support_ratio=1.0 if value >= 0.55 else 0.0,
        component_scores={},
    )


def test_classify_verification_uses_calibrated_support_tiers():
    assert classify_verification(_score(0.71)).label == "VERIFIED"
    assert classify_verification(_score(0.55)).label == "SUPPORTED"
    assert classify_verification(_score(0.40)).label == "POSSIBLE_SUPPORT"
    assert classify_verification(_score(0.39)).label == "UNSUPPORTED"


def test_classify_verification_prioritizes_contradiction_and_no_evidence():
    assert classify_verification(_score(0.90, contradicted=True)).label == "CONTRADICTED"
    assert classify_verification(_score(0.90, has_chunk=False)).label == "NO_EVIDENCE"
