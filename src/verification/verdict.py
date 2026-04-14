"""Verdict classification helpers for aggregated verification scores."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import VERIFICATION
from src.verification.nli_verifier import AggregatedScore


@dataclass(frozen=True)
class VerificationClassification:
    label: str
    explanation: str


def classify_verification(score: AggregatedScore) -> VerificationClassification:
    """Map an aggregated verification score to an API-friendly verdict."""
    contradiction_signal = float(
        score.component_scores.get(
            "max_contradiction",
            score.component_scores.get("contradiction_max", 0.0),
        )
    )
    if score.is_contradicted and contradiction_signal >= VERIFICATION.threshold_contradicted:
        return VerificationClassification(
            label="CONTRADICTED",
            explanation="Retrieved evidence materially contradicts the claim.",
        )

    if score.final_score >= VERIFICATION.threshold_verified:
        return VerificationClassification(
            label="VERIFIED",
            explanation="Retrieved evidence strongly supports the claim.",
        )

    if score.final_score >= VERIFICATION.threshold_supported:
        return VerificationClassification(
            label="SUPPORTED",
            explanation="Retrieved evidence supports the claim.",
        )

    if score.final_score >= VERIFICATION.threshold_weak:
        return VerificationClassification(
            label="POSSIBLE_SUPPORT",
            explanation="Retrieved evidence is directionally supportive but not decisive.",
        )

    return VerificationClassification(
        label="UNSUPPORTED",
        explanation="Retrieved evidence does not sufficiently support the claim.",
    )
