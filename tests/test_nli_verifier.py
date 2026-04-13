"""Tests for NLI verification."""

from dataclasses import dataclass

import pytest

from src.ingestion.document import LegalChunk
from src.verification.nli_verifier import NLIVerifier


pytestmark = pytest.mark.smoke


@dataclass(frozen=True)
class _Claim:
    text: str


class _FakeNLIVerifier(NLIVerifier):
    def __init__(self, score_map):
        super().__init__(device="cpu")
        self.score_map = score_map
        self.seen_pairs = []

    def _predict_pairs(self, pairs):
        self.seen_pairs.extend(pairs)
        return [self.score_map[pair] for pair in pairs]


def _chunk(chunk_id: str, text: str, court_level: str, idx: int) -> LegalChunk:
    return LegalChunk(
        id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        text=text,
        chunk_index=idx,
        doc_type="case",
        court_level=court_level,
    )


def test_verify_claim_uses_authority_weighted_best_chunk():
    claim = _Claim("Police needed a warrant.")
    scotus_chunk = _chunk("a", "SCOTUS says warrant required.", "scotus", 0)
    district_chunk = _chunk("b", "District court says maybe not.", "district", 1)

    verifier = _FakeNLIVerifier(
        {
            (scotus_chunk.text, claim.text): {
                "entailment": 0.90,
                "neutral": 0.08,
                "contradiction": 0.02,
            },
            (district_chunk.text, claim.text): {
                "entailment": 0.95,
                "neutral": 0.03,
                "contradiction": 0.02,
            },
        }
    )

    result = verifier.verify_claim(claim, [scotus_chunk, district_chunk])

    assert result.best_chunk == scotus_chunk
    assert result.best_chunk_idx == 0
    assert result.support_ratio == 1.0
    assert result.is_contradicted is False
    assert result.final_score > 0.0


def test_verify_claims_batch_builds_all_claim_chunk_pairs():
    claim_a = _Claim("A")
    claim_b = _Claim("B")
    chunk_a = _chunk("a", "Chunk A", "scotus", 0)
    chunk_b = _chunk("b", "Chunk B", "circuit", 1)

    verifier = _FakeNLIVerifier(
        {
            (chunk_a.text, claim_a.text): {"entailment": 0.8, "neutral": 0.1, "contradiction": 0.1},
            (chunk_b.text, claim_a.text): {"entailment": 0.3, "neutral": 0.5, "contradiction": 0.2},
            (chunk_a.text, claim_b.text): {"entailment": 0.2, "neutral": 0.5, "contradiction": 0.3},
            (chunk_b.text, claim_b.text): {"entailment": 0.85, "neutral": 0.1, "contradiction": 0.05},
        }
    )

    results = verifier.verify_claims_batch([claim_a, claim_b], [chunk_a, chunk_b])

    assert len(results) == 2
    assert len(verifier.seen_pairs) == 4
    assert results[0].best_chunk == chunk_a
    assert results[1].best_chunk == chunk_b


def test_contradiction_penalty_flags_contradicted_claim():
    claim = _Claim("The motion was granted.")
    chunk = _chunk("a", "The motion was denied.", "scotus", 0)

    verifier = _FakeNLIVerifier(
        {
            (chunk.text, claim.text): {
                "entailment": 0.15,
                "neutral": 0.05,
                "contradiction": 0.80,
            }
        }
    )

    result = verifier.verify_claim(claim, [chunk])

    assert result.is_contradicted is True
    assert result.component_scores["contra_penalty"] == pytest.approx(0.80)
    assert result.final_score < 0.20


def test_empty_chunk_list_returns_empty_result():
    verifier = _FakeNLIVerifier({})

    result = verifier.verify_claim(_Claim("Anything"), [])

    assert result.final_score == 0.0
    assert result.best_chunk is None
    assert result.best_chunk_idx == -1
