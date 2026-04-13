"""Tests for citation verification."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from src.ingestion.document import LegalChunk
from src.verification.citation_verifier import (
    chunk_text,
    classify_score,
    extract_citations,
    extract_claimed_proposition,
    verify_citation_claim,
    verify_claim_against_text,
    verify_claims,
)
from src.verification.claim_decomposer import Claim, SpanRef, decompose_document


pytestmark = pytest.mark.smoke
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "artifacts" / "test_reports"


class _FakeModel:
    def __init__(self) -> None:
        self.vocabulary = [
            "court",
            "held",
            "defendant",
            "violated",
            "contract",
            "payment",
            "terms",
            "goods",
            "failed",
            "deliver",
            "deadline",
            "obligations",
            "weather",
            "forecast",
            "sunny",
            "today",
            "late",
            "miranda",
            "arizona",
            "warnings",
            "required",
            "custodial",
            "interrogation",
            "roe",
            "privacy",
            "access",
            "protected",
            "computer",
            "unauthorized",
            "prohibited",
        ]

    def encode(self, inputs, convert_to_tensor=True):
        if isinstance(inputs, str):
            return self._vectorize(inputs)
        return np.vstack([self._vectorize(item) for item in inputs])

    def _vectorize(self, text: str) -> np.ndarray:
        tokens = set(re.findall(r"\b\w+\b", text.lower()))
        return np.array([1.0 if token in tokens else 0.0 for token in self.vocabulary], dtype=float)


def _claim(
    text: str,
    *,
    claim_type: str = "fact",
    certainty: str = "found",
) -> Claim:
    return Claim(
        claim_id="clm_test",
        text=text,
        claim_type=claim_type,
        source="court",
        certainty=certainty,
        doc_section="body",
        span=SpanRef(
            doc_id="doc_test",
            para_id=0,
            sent_id=1,
            start_char=0,
            end_char=len(text),
        ),
    )


def _chunk(
    chunk_id: str,
    text: str,
    *,
    citation: str | None = None,
    doc_type: str = "case",
    doc_id: str | None = None,
) -> LegalChunk:
    return LegalChunk(
        id=chunk_id,
        doc_id=doc_id or f"doc_{chunk_id}",
        text=text,
        chunk_index=0,
        doc_type=doc_type,
        court_level="scotus",
        citation=citation,
    )


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _thresholds_for_claim(claim: Claim) -> dict[str, float]:
    base_high = 0.75
    base_mid = 0.55

    if claim.certainty == "alleged":
        base_high -= 0.10
    elif claim.certainty == "found":
        base_high += 0.05

    if claim.claim_type == "holding":
        base_high += 0.05
    elif claim.claim_type == "fact":
        base_mid -= 0.05

    return {
        "supported_at_or_above": round(base_high, 3),
        "partial_at_or_above": round(base_mid, 3),
    }


def _write_report(filename: str, payload: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / filename
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def test_chunk_text_splits_into_fixed_size_word_groups():
    chunks = chunk_text("one two three four five six", chunk_size=2)
    assert chunks == ["one two", "three four", "five six"]


def test_classify_score_respects_claim_metadata_thresholds():
    alleged_fact = _claim("Plaintiff alleges late delivery.", certainty="alleged", claim_type="fact")
    found_holding = _claim("The court held the contract was violated.", claim_type="holding")

    assert classify_score(0.66, alleged_fact) == "SUPPORTED"
    assert classify_score(0.79, found_holding) == "PARTIAL"


def test_verify_claim_against_text_returns_best_matching_chunk():
    claim = _claim("The defendant violated the contract.", certainty="alleged")
    source_text = "defendant violated contract payment terms unrelated weather forecast sunny today"

    result = verify_claim_against_text(
        claim,
        source_text,
        model=_FakeModel(),
        chunk_size=5,
    )

    assert result["verdict"] == "SUPPORTED"
    assert result["evidence"] == "defendant violated contract payment terms"
    assert result["confidence"] == pytest.approx(0.775, abs=0.001)


def test_verify_claim_against_text_handles_empty_source_text():
    claim = _claim("Any claim.")

    result = verify_claim_against_text(claim, "   ", model=_FakeModel())

    assert result["verdict"] == "ERROR"
    assert result["reason"] == "No source text"


def test_verify_claims_accepts_decomposed_claims_batch():
    document = {
        "id": "test_case",
        "full_text": (
            "The court held the defendant violated the contract. "
            "The defendant failed to deliver goods."
        ),
    }
    claims = decompose_document(document)
    source_text = (
        "court held defendant violated contract agreed obligations "
        "defendant failed deliver goods before deadline"
    )
    model = _FakeModel()
    chunk_size = 6
    chunks = chunk_text(source_text, chunk_size=chunk_size)
    chunk_vectors = model.encode(chunks, convert_to_tensor=True)

    results = verify_claims(
        claims,
        source_text,
        model=model,
        chunk_size=chunk_size,
    )

    report = {
        "test_name": "test_verify_claims_accepts_decomposed_claims_batch",
        "document": document,
        "source_text": source_text,
        "chunk_size": chunk_size,
        "source_chunks": [
            {
                "chunk_index": index,
                "text": chunk,
                "word_count": len(chunk.split()),
            }
            for index, chunk in enumerate(chunks)
        ],
        "claims": [],
    }

    for claim, result in zip(claims, results):
        claim_vector = model.encode(claim.text, convert_to_tensor=True)
        chunk_scores = [
            {
                "chunk_index": index,
                "chunk_text": chunk,
                "score": round(_cosine_similarity(claim_vector, chunk_vector), 3),
            }
            for index, (chunk, chunk_vector) in enumerate(zip(chunks, chunk_vectors))
        ]
        best_match = max(chunk_scores, key=lambda item: item["score"])
        report["claims"].append(
            {
                "claim_id": claim.claim_id,
                "text": claim.text,
                "claim_type": claim.claim_type,
                "certainty": claim.certainty,
                "source": claim.source,
                "thresholds": _thresholds_for_claim(claim),
                "chunk_scores": chunk_scores,
                "best_match_before_classification": best_match,
                "final_result": result,
            }
        )

    report["summary"] = {
        "claim_count": len(claims),
        "result_count": len(results),
        "all_verdicts": [result["verdict"] for result in results],
    }
    report_path = _write_report("semantic_support_trace.json", report)

    assert len(results) == len(claims) == 2
    assert [result["verdict"] for result in results] == ["SUPPORTED", "SUPPORTED"]
    assert report_path.exists()


def test_extract_citations_finds_case_and_statutory_references():
    text = (
        "Miranda v. Arizona, 384 U.S. 436 (1966), requires warnings. "
        "18 U.S.C. § 1030 prohibits unauthorized access to a protected computer."
    )

    citations = extract_citations(text)

    assert [citation.raw_text for citation in citations] == [
        "384 U.S. 436",
        "18 U.S.C. § 1030",
    ]
    assert citations[0].case_name == "Miranda v. Arizona"
    assert citations[0].citation_type == "case"
    assert citations[1].citation_type == "statute"


def test_verify_citation_claim_matches_authority_before_scoring_support():
    claim = _claim(
        "Miranda v. Arizona, 384 U.S. 436 (1966), held that Miranda warnings are required during custodial interrogation.",
        claim_type="citation",
    )
    source_documents = [
        {
            "doc_id": "miranda_v_arizona_384_us_436",
            "authority_name": "Miranda v. Arizona",
            "citation": "384 U.S. 436",
            "supports": "Miranda warnings are required during custodial interrogation.",
        },
        {
            "doc_id": "roe_v_wade_410_us_113",
            "authority_name": "Roe v. Wade",
            "citation": "410 U.S. 113",
            "supports": "A privacy right in abortion decisions.",
        },
    ]
    source_chunks = [
        _chunk(
            "miranda",
            "The Supreme Court held that Miranda warnings are required during custodial interrogation.",
            citation="384 U.S. 436",
            doc_id="miranda_v_arizona_384_us_436",
        ),
        _chunk(
            "roe",
            "The Court recognized a privacy right in abortion decisions.",
            citation="410 U.S. 113",
            doc_id="roe_v_wade_410_us_113",
        ),
        _chunk(
            "uncited",
            "Miranda warnings are required during custodial interrogation.",
            citation=None,
            doc_id="uncited_secondary_excerpt",
        ),
    ]

    result = verify_citation_claim(claim, source_chunks, model=_FakeModel())
    proposition = extract_claimed_proposition(claim.text)
    citations = extract_citations(claim.text)
    matched_document = next(
        (
            document
            for document in source_documents
            if document["doc_id"] == result["matched_doc_id"]
        ),
        None,
    )
    report = {
        "test_name": "test_verify_citation_claim_matches_authority_before_scoring_support",
        "claim": {
            "text": claim.text,
            "claim_type": claim.claim_type,
            "parsed_citations": [citation.to_dict() for citation in citations],
            "claimed_proposition": proposition,
        },
        "source_documents": source_documents,
        "source_chunks": [
            {
                "id": chunk.id,
                "doc_id": chunk.doc_id,
                "citation": chunk.citation,
                "text_preview": chunk.text[:200],
            }
            for chunk in source_chunks
        ],
        "resolved_supporting_document": matched_document,
        "verification_result": result,
    }
    report_path = _write_report("citation_verifier_trace.json", report)

    assert result["citation_exists"] is True
    assert result["citation_format_valid"] is True
    assert result["authority_found"] is True
    assert result["matched_doc_id"] == "miranda_v_arizona_384_us_436"
    assert result["matched_chunk_id"] == "miranda"
    assert result["matched_citation"] == "384 U.S. 436"
    assert result["candidate_chunk_ids"] == ["miranda"]
    assert result["claimed_proposition"] == proposition
    assert result["verdict"] == "VERIFIED"
    assert result["confidence"] > 0.9
    assert matched_document is not None
    assert matched_document["authority_name"] == "Miranda v. Arizona"
    assert report_path.exists()


def test_verify_citation_claim_rejects_semantically_supported_text_with_wrong_citation():
    claim = _claim(
        "Roe v. Wade, 410 U.S. 113 (1973), held that Miranda warnings are required during custodial interrogation.",
        claim_type="citation",
    )
    source_chunks = [
        _chunk(
            "miranda",
            "The Supreme Court held that Miranda warnings are required during custodial interrogation.",
            citation="384 U.S. 436",
            doc_id="miranda_v_arizona_384_us_436",
        ),
    ]

    result = verify_citation_claim(claim, source_chunks, model=_FakeModel())

    assert result["citation_exists"] is False
    assert result["citation_format_valid"] is True
    assert result["authority_found"] is False
    assert result["matched_doc_id"] is None
    assert result["matched_chunk_id"] is None
    assert result["verdict"] == "CITATION_ERROR"
