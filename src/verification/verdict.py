"""
Algorithm 4: Calibrated classification for final verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import VERIFICATION
from src.verification.nli_verifier import AggregatedScore


@dataclass(frozen=True)
class VerdictClassification:
    """Human-readable claim classification derived from aggregate NLI scores."""

    label: str
    explanation: str


def classify_verification(verdict: AggregatedScore) -> VerdictClassification:
    """Classify a claim verification score using calibrated thresholds."""
    if verdict.best_chunk is None or verdict.best_chunk_idx < 0:
        return VerdictClassification(
            label="NO_EVIDENCE",
            explanation="No retrieved evidence was available for this claim.",
        )

    if verdict.is_contradicted:
        return VerdictClassification(
            label="CONTRADICTED",
            explanation="The strongest evidence contains a contradiction signal above threshold.",
        )

    score = verdict.final_score
    if score >= VERIFICATION.threshold_verified:
        return VerdictClassification(
            label="VERIFIED",
            explanation="The claim is strongly and explicitly supported by retrieved evidence.",
        )
    if score >= VERIFICATION.threshold_supported:
        return VerdictClassification(
            label="SUPPORTED",
            explanation="The claim is supported by retrieved evidence.",
        )
    if score >= VERIFICATION.threshold_possible_support:
        return VerdictClassification(
            label="POSSIBLE_SUPPORT",
            explanation=(
                "The claim has possible support, but it falls below the threshold "
                "for automatic confirmation."
            ),
        )
    return VerdictClassification(
        label="UNSUPPORTED",
        explanation="The retrieved evidence does not sufficiently support the claim.",
    )
