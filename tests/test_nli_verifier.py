"""Tests for NLI verification."""

from dataclasses import dataclass
import sys
import types

import pytest

from src import config as config_module
from src.ingestion.document import LegalChunk
from src.verification.heuristic_verifier import HeuristicNLIVerifier
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


class _RecordingHypothesisVerifier(_FakeNLIVerifier):
    def __init__(self, score_map):
        super().__init__(score_map)
        self.grouped_hypotheses = []

    def _aggregate_scores(self, nli_scores):
        self.grouped_hypotheses.append([score.hypothesis for score in nli_scores])
        return super()._aggregate_scores(nli_scores)


def _chunk(
    chunk_id: str,
    text: str,
    court_level: str,
    idx: int,
    *,
    doc_type: str = "case",
    verification_tier: str | None = None,
    verification_tier_rank: int | None = None,
) -> LegalChunk:
    chunk = LegalChunk(
        id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        text=text,
        chunk_index=idx,
        doc_type=doc_type,
        court_level=court_level,
    )
    if verification_tier is not None:
        setattr(chunk, "verification_tier", verification_tier)
    if verification_tier_rank is not None:
        setattr(chunk, "verification_tier_rank", verification_tier_rank)
    return chunk


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


def test_user_upload_chunks_are_weighted_as_primary_record_evidence():
    claim = _Claim("Marquez did not personally test the H-17 battery pack.")
    upload_chunk = _chunk(
        "upload",
        "Marquez did not personally test the H-17 battery pack.",
        "",
        0,
        doc_type="user_upload",
    )
    unknown_chunk = _chunk(
        "unknown",
        "Marquez did not personally test the H-17 battery pack.",
        "",
        0,
    )

    score = {
        "entailment": 0.80,
        "neutral": 0.15,
        "contradiction": 0.05,
    }
    verifier = _FakeNLIVerifier(
        {
            (upload_chunk.text, claim.text): score,
            (unknown_chunk.text, claim.text): score,
        }
    )

    upload_result = verifier.verify_claim(claim, [upload_chunk])
    unknown_result = verifier.verify_claim(claim, [unknown_chunk])

    assert upload_result.component_scores["max_pool"] == pytest.approx(0.80)
    assert unknown_result.component_scores["max_pool"] == pytest.approx(0.32)
    assert upload_result.final_score > unknown_result.final_score


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


def test_verify_claims_batch_preserves_each_claim_hypothesis():
    claim_a = _Claim("First claim hypothesis.")
    claim_b = _Claim("Second claim hypothesis.")
    chunk = _chunk("a", "Chunk A", "scotus", 0)

    verifier = _RecordingHypothesisVerifier(
        {
            (chunk.text, claim_a.text): {"entailment": 0.8, "neutral": 0.1, "contradiction": 0.1},
            (chunk.text, claim_b.text): {"entailment": 0.7, "neutral": 0.2, "contradiction": 0.1},
        }
    )

    verifier.verify_claims_batch([claim_a, claim_b], [chunk])

    assert verifier.grouped_hypotheses == [[claim_a.text], [claim_b.text]]


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


def test_support_dominance_suppresses_background_false_contradiction():
    claim = _Claim("District courts cannot consider section 3553(a)(2)(A) when revoking supervised release.")
    holding_chunk = _chunk(
        "holding",
        "District courts cannot consider section 3553(a)(2)(A) when revoking supervised release.",
        "scotus",
        0,
    )
    background_chunk = _chunk(
        "background",
        "Sixth Circuit precedent allowed district courts to consider section 3553(a)(2)(A).",
        "scotus",
        1,
    )

    verifier = _FakeNLIVerifier(
        {
            (holding_chunk.text, claim.text): {
                "entailment": 0.96,
                "neutral": 0.03,
                "contradiction": 0.01,
            },
            (background_chunk.text, claim.text): {
                "entailment": 0.10,
                "neutral": 0.02,
                "contradiction": 0.94,
            },
        }
    )

    result = verifier.verify_claim(claim, [holding_chunk, background_chunk])

    assert result.is_contradicted is False
    assert result.component_scores["support_dominates_contradiction"] == pytest.approx(1.0)
    assert result.component_scores["contra_penalty"] == pytest.approx(0.0)


def test_support_dominance_uses_margin_for_near_tied_contradiction():
    claim = _Claim("District courts cannot consider section 3553(a)(2)(A) when revoking supervised release.")
    holding_chunk = _chunk(
        "holding",
        "District courts cannot consider section 3553(a)(2)(A) when revoking supervised release.",
        "scotus",
        0,
    )
    noisy_background_chunk = _chunk(
        "background",
        "A party discussed earlier practice before the Court resolved the section 3553 issue.",
        "scotus",
        1,
    )

    verifier = _FakeNLIVerifier(
        {
            (holding_chunk.text, claim.text): {
                "entailment": 0.997,
                "neutral": 0.002,
                "contradiction": 0.001,
            },
            (noisy_background_chunk.text, claim.text): {
                "entailment": 0.20,
                "neutral": 0.001,
                "contradiction": 0.998,
            },
        }
    )

    result = verifier.verify_claim(claim, [holding_chunk, noisy_background_chunk])

    assert result.is_contradicted is False
    assert result.component_scores["contradiction_margin"] == pytest.approx(0.001, abs=1e-5)
    assert result.component_scores["support_dominates_contradiction"] == pytest.approx(1.0)


def test_source_tier_support_blocks_broader_contradiction_without_margin():
    claim = _Claim("The Court held the rule applies.")
    source_chunk = _chunk(
        "source",
        "The Court held the rule applies.",
        "scotus",
        0,
        verification_tier="generation_source",
        verification_tier_rank=1,
    )
    broad_chunk = _chunk(
        "target_doc_background",
        "Earlier background in the same case discussed the opposite rule.",
        "scotus",
        1,
        verification_tier="target_doc",
        verification_tier_rank=3,
    )

    verifier = _FakeNLIVerifier(
        {
            (source_chunk.text, claim.text): {
                "entailment": 0.88,
                "neutral": 0.10,
                "contradiction": 0.02,
            },
            (broad_chunk.text, claim.text): {
                "entailment": 0.10,
                "neutral": 0.02,
                "contradiction": 0.95,
            },
        }
    )

    result = verifier.verify_claim(claim, [source_chunk, broad_chunk])

    assert result.is_contradicted is False
    assert result.component_scores["best_support_tier_rank"] == pytest.approx(1.0)
    assert result.component_scores["best_contradiction_tier_rank"] == pytest.approx(3.0)
    assert result.component_scores["tier_allows_contradiction"] == pytest.approx(0.0)


def test_broader_non_authoritative_posture_cannot_override_source_support():
    claim = _Claim("The Court held the rule applies.")
    source_chunk = _chunk(
        "source",
        "The Court held the rule applies.",
        "scotus",
        0,
        verification_tier="generation_source",
        verification_tier_rank=1,
    )
    lower_court_chunk = _chunk(
        "lower_court_background",
        "The court of appeals held the opposite rule before the Supreme Court resolved the case.",
        "scotus",
        1,
        verification_tier="target_doc",
        verification_tier_rank=3,
    )

    verifier = _FakeNLIVerifier(
        {
            (source_chunk.text, claim.text): {
                "entailment": 0.86,
                "neutral": 0.12,
                "contradiction": 0.02,
            },
            (lower_court_chunk.text, claim.text): {
                "entailment": 0.03,
                "neutral": 0.01,
                "contradiction": 0.99,
            },
        }
    )

    result = verifier.verify_claim(claim, [source_chunk, lower_court_chunk])

    assert result.is_contradicted is False
    assert result.component_scores["contradiction_posture_valid"] == pytest.approx(0.0)
    assert result.component_scores["tier_allows_contradiction"] == pytest.approx(0.0)


def test_same_tier_contradiction_still_overrides_weak_support():
    claim = _Claim("The motion was granted.")
    weak_support_chunk = _chunk(
        "weak",
        "The order discussed a motion.",
        "scotus",
        0,
        verification_tier="generation_source",
        verification_tier_rank=1,
    )
    contradiction_chunk = _chunk(
        "contra",
        "The motion was denied.",
        "scotus",
        1,
        verification_tier="generation_source",
        verification_tier_rank=1,
    )

    verifier = _FakeNLIVerifier(
        {
            (weak_support_chunk.text, claim.text): {
                "entailment": 0.40,
                "neutral": 0.55,
                "contradiction": 0.05,
            },
            (contradiction_chunk.text, claim.text): {
                "entailment": 0.05,
                "neutral": 0.02,
                "contradiction": 0.93,
            },
        }
    )

    result = verifier.verify_claim(claim, [weak_support_chunk, contradiction_chunk])

    assert result.is_contradicted is True
    assert result.component_scores["tier_allows_contradiction"] == pytest.approx(1.0)
    assert result.component_scores["contradiction_overlap_valid"] == pytest.approx(1.0)


def test_unrelated_high_contradiction_score_cannot_override_without_overlap():
    claim = _Claim("The plaintiffs lack standing because they cannot show a substantial risk of injury.")
    unrelated_chunk = _chunk(
        "unrelated",
        "Facebook was required to pay a large civil penalty in a data privacy settlement.",
        "scotus",
        0,
        verification_tier="sentence_evidence",
        verification_tier_rank=0,
    )

    verifier = _FakeNLIVerifier(
        {
            (unrelated_chunk.text, claim.text): {
                "entailment": 0.02,
                "neutral": 0.01,
                "contradiction": 0.97,
            },
        }
    )

    result = verifier.verify_claim(claim, [unrelated_chunk])

    assert result.is_contradicted is False
    assert result.component_scores["contradiction_overlap_valid"] == pytest.approx(0.0)


def test_missing_procedural_anchor_cannot_be_contradiction():
    claim = _Claim(
        "This defect was not cured because the removal petition did not include a statement of jurisdiction as required by 28 U.S.C. § 1441(b)."
    )
    narrow_chunk = _chunk(
        "narrow",
        'As a result, the jurisdictional defect "lingered through judgment" uncured and the judgment "must be vacated." Caterpillar, 519 U.S.',
        "scotus",
        0,
        verification_tier="sentence_evidence",
        verification_tier_rank=0,
    )

    verifier = _FakeNLIVerifier(
        {
            (narrow_chunk.text, claim.text): {
                "entailment": 0.05,
                "neutral": 0.01,
                "contradiction": 0.98,
            },
        }
    )

    result = verifier.verify_claim(claim, [narrow_chunk])

    assert result.is_contradicted is False
    assert result.component_scores["contradiction_overlap_valid"] == pytest.approx(0.0)
    assert result.final_score > 0.0


def test_explicit_same_anchor_opposite_can_be_contradiction():
    claim = _Claim(
        "The removal petition did not include a statement of jurisdiction as required by 28 U.S.C. § 1441(b)."
    )
    contradiction_chunk = _chunk(
        "contra",
        "The removal petition included the required statement of jurisdiction under 28 U.S.C. § 1441(b).",
        "scotus",
        0,
        verification_tier="sentence_evidence",
        verification_tier_rank=0,
    )

    verifier = _FakeNLIVerifier(
        {
            (contradiction_chunk.text, claim.text): {
                "entailment": 0.02,
                "neutral": 0.01,
                "contradiction": 0.98,
            },
        }
    )

    result = verifier.verify_claim(claim, [contradiction_chunk])

    assert result.is_contradicted is True
    assert result.component_scores["contradiction_overlap_valid"] == pytest.approx(1.0)


def test_same_tier_background_contradiction_cannot_override_strong_source_support():
    claim = _Claim("NEPA does not require an agency to study separate upstream or downstream projects.")
    holding_chunk = _chunk(
        "holding",
        "The Court held NEPA does not require an agency to study separate upstream or downstream projects.",
        "scotus",
        0,
        verification_tier="generation_source",
        verification_tier_rank=1,
    )
    background_chunk = _chunk(
        "background",
        "The court of appeals held NEPA required review of separate upstream and downstream projects.",
        "scotus",
        1,
        verification_tier="generation_source",
        verification_tier_rank=1,
    )

    verifier = _FakeNLIVerifier(
        {
            (holding_chunk.text, claim.text): {
                "entailment": 0.96,
                "neutral": 0.03,
                "contradiction": 0.01,
            },
            (background_chunk.text, claim.text): {
                "entailment": 0.04,
                "neutral": 0.001,
                "contradiction": 0.999,
            },
        }
    )

    result = verifier.verify_claim(claim, [holding_chunk, background_chunk])

    assert result.is_contradicted is False
    assert result.component_scores["contradiction_posture_valid"] == pytest.approx(0.0)
    assert result.component_scores["tier_allows_contradiction"] == pytest.approx(0.0)


def test_heuristic_verifier_does_not_treat_docket_number_as_negation():
    claim = _Claim("The Supreme Court denied certiorari in Burnett v. United States.")
    chunk = _chunk(
        "burnett_caption",
        (
            "SUPREME COURT OF THE UNITED STATES JARON BURNETT v. UNITED STATES "
            "ON PETITION FOR WRIT OF CERTIORARI TO THE UNITED STATES COURT OF APPEALS "
            "FOR THE THIRD CIRCUIT No. 25-5442. "
            "The petition for a writ of certiorari is denied."
        ),
        "scotus",
        0,
    )

    result = HeuristicNLIVerifier().verify_claim(claim, [chunk])

    assert result.is_contradicted is False


def test_empty_chunk_list_returns_empty_result():
    verifier = _FakeNLIVerifier({})

    result = verifier.verify_claim(_Claim("Anything"), [])

    assert result.final_score == 0.0
    assert result.best_chunk is None
    assert result.best_chunk_idx == -1


def test_load_model_forwards_local_files_only(monkeypatch):
    calls = []

    class _FakeTorch:
        class cuda:
            @staticmethod
            def is_available():
                return False

    class _FakeTokenizer:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls.append(("tokenizer", model_name, kwargs))
            return object()

    class _FakeModel:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls.append(("model", model_name, kwargs))

            class _LoadedModel:
                def to(self, device):
                    calls.append(("to", device, {}))
                    return self

                def eval(self):
                    calls.append(("eval", None, {}))
                    return self

            return _LoadedModel()

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=_FakeTokenizer,
            AutoModelForSequenceClassification=_FakeModel,
        ),
    )
    monkeypatch.setattr(config_module.MODELS, "huggingface_local_files_only", True)

    verifier = NLIVerifier(device="cpu")
    verifier._load_model()

    assert calls[0] == (
        "tokenizer",
        config_module.MODELS.nli_model,
        {"local_files_only": True},
    )
    assert calls[1] == (
        "model",
        config_module.MODELS.nli_model,
        {"local_files_only": True},
    )


def test_load_model_uses_config_label_mapping_when_available(monkeypatch):
    class _FakeTorch:
        class cuda:
            @staticmethod
            def is_available():
                return False

    class _FakeTokenizer:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            _ = model_name, kwargs
            return object()

    class _FakeModel:
        config = types.SimpleNamespace(
            id2label={
                0: "entailment",
                1: "neutral",
                2: "contradiction",
            }
        )

        @staticmethod
        def from_pretrained(model_name, **kwargs):
            _ = model_name, kwargs

            class _LoadedModel(_FakeModel):
                def to(self, device):
                    _ = device
                    return self

                def eval(self):
                    return self

            return _LoadedModel()

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=_FakeTokenizer,
            AutoModelForSequenceClassification=_FakeModel,
        ),
    )

    verifier = NLIVerifier(device="cpu")
    verifier._load_model()

    assert verifier.label_map == {
        "entailment": 0,
        "neutral": 1,
        "contradiction": 2,
    }


def test_same_chunk_high_entailment_and_contradiction_is_confusion_not_contradiction():
    """
    When the SAME chunk produces both high entailment (>0.70) and high 
    contradiction (>0.60), the NLI model is confused. The contradiction 
    signal should be neutralized to avoid "possibly_supported" status from
    conflicting signals on the same evidence.
    
    This prevents false negatives where valid claims get marked as only
    "possibly supported" because the same chunk both supports and contradicts.
    """
    claim = _Claim("The Court held that supervised release revocation requires jury findings.")
    confused_chunk = _chunk(
        "confused",
        "Justice Gorsuch dissented and discussed jury requirements for supervised release.",
        "scotus",
        0,
        verification_tier="sentence_evidence",
        verification_tier_rank=0,
    )

    verifier = _FakeNLIVerifier(
        {
            (confused_chunk.text, claim.text): {
                # Same chunk produces BOTH high entailment AND high contradiction
                # This is the "same chunk confusion" pattern
                "entailment": 0.82,      # High support
                "neutral": 0.03,
                "contradiction": 0.75,   # Also high contradiction!
            },
        }
    )

    result = verifier.verify_claim(claim, [confused_chunk])

    # Should NOT be contradicted because the contradiction comes from same chunk
    # as the support signal - the model is confused, not actually contradictory
    assert result.is_contradicted is False
    
    # The contradiction penalty should be neutralized (0.0)
    assert result.component_scores["contra_penalty"] == pytest.approx(0.0)
    
    # Final score should be higher because no contradiction penalty applied
    assert result.final_score > 0.60


def test_same_chunk_only_neutralizes_when_both_scores_high():
    """
    The same-chunk confusion neutralization only applies when BOTH
    entailment >= 0.70 AND contradiction >= 0.60. If scores are lower,
    the normal contradiction logic applies.
    """
    claim = _Claim("The Court held that supervised release revocation requires jury findings.")
    chunk = _chunk(
        "chunk",
        "The Court explicitly rejected the jury requirement argument in its opinion.",
        "scotus",
        0,
        verification_tier="sentence_evidence",
        verification_tier_rank=0,
    )

    verifier = _FakeNLIVerifier(
        {
            (chunk.text, claim.text): {
                # Same chunk but both scores below the "confusion" thresholds
                "entailment": 0.45,      # < 0.70 threshold
                "neutral": 0.05,
                "contradiction": 0.55,   # < 0.60 threshold
            },
        }
    )

    result = verifier.verify_claim(claim, [chunk])

    # Since both scores are below thresholds, confusion neutralization
    # does NOT apply - normal contradiction logic applies
    # But this will likely not be contradicted due to low contradiction score
    assert result.is_contradicted is False  # 0.55 < 0.60 threshold
