"""Tests for verification verdict classification."""

from __future__ import annotations

from src.verification.nli_verifier import AggregatedScore
from src.verification.verdict import classify_verification


def _score(
    final_score: float,
    *,
    is_contradicted: bool = False,
    contradiction_value: float = 0.0,
) -> AggregatedScore:
    return AggregatedScore(
        final_score=final_score,
        is_contradicted=is_contradicted,
        best_chunk_idx=-1,
        best_chunk=None,
        support_ratio=0.0,
        component_scores={"max_contradiction": contradiction_value},
    )


def test_classify_verification_uses_calibrated_threshold_bands():
    assert classify_verification(_score(0.72)).label == "VERIFIED"
    assert classify_verification(_score(0.60)).label == "SUPPORTED"
    assert classify_verification(_score(0.52)).label == "POSSIBLE_SUPPORT"
    assert classify_verification(_score(0.49)).label == "UNSUPPORTED"


def test_classify_verification_marks_contradictions_from_aggregated_score():
    classification = classify_verification(
        _score(0.61, is_contradicted=True, contradiction_value=0.72)
    )

    assert classification.label == "CONTRADICTED"
