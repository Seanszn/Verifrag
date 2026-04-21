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
