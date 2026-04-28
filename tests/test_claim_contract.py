"""Tests for the stable client-facing claim contract."""

from __future__ import annotations

from src.verification.claim_contract import normalize_claims_for_frontend, support_level_from_verdict


def test_support_level_from_verdict_collapses_internal_labels():
    assert support_level_from_verdict("VERIFIED") == "supported"
    assert support_level_from_verdict("SUPPORTED") == "supported"
    assert support_level_from_verdict("POSSIBLE_SUPPORT") == "possibly_supported"
    assert support_level_from_verdict("UNSUPPORTED") == "unsupported"
    assert support_level_from_verdict("CONTRADICTED") == "unsupported"
    assert support_level_from_verdict("NO_EVIDENCE") == "unsupported"


def test_normalize_claims_for_frontend_builds_annotation_and_links():
    claims = [
        {
            "claim_id": "claim-1",
            "text": "Miranda warnings are required.",
            "span": {
                "doc_id": "assistant_response",
                "para_id": 0,
                "sent_id": 1,
                "start_char": 0,
                "end_char": 31,
            },
            "verification": {
                "verdict": "SUPPORTED",
                "verdict_explanation": "Retrieved evidence supports the claim.",
                "best_chunk": {
                    "id": "chunk-miranda",
                    "doc_id": "doc-miranda",
                    "citation": "384 U.S. 436",
                    "text": "Miranda requires warnings before custodial interrogation.",
                },
                "best_supporting_chunk": {
                    "id": "chunk-miranda",
                    "doc_id": "doc-miranda",
                    "citation": "384 U.S. 436",
                    "text": "Miranda requires warnings before custodial interrogation.",
                },
                "best_supporting_score": 0.91,
                "best_contradicting_chunk": {
                    "id": "chunk-example",
                    "doc_id": "doc-example",
                    "citation": "123 Example 456",
                    "text_preview": "Example narrows the rule.",
                },
                "best_contradiction_score": 0.22,
            },
        }
    ]
    citations = [
        {"id": "chunk-miranda", "doc_id": "doc-miranda", "citation": "384 U.S. 436"},
        {"id": "chunk-example", "doc_id": "doc-example", "citation": "123 Example 456"},
    ]

    normalized_claims, links = normalize_claims_for_frontend(claims, citations=citations)

    assert len(links) == 2
    assert [link["relationship"] for link in links] == ["supporting", "contradicting"]
    assert normalized_claims[0]["annotation"]["support_level"] == "supported"
    assert normalized_claims[0]["annotation"]["response_span"]["start_char"] == 0
    assert [item["relationship"] for item in normalized_claims[0]["annotation"]["evidence"]] == [
        "supporting",
        "contradicting",
    ]
    assert normalized_claims[0]["annotation"]["evidence"][0]["evidence_quote"] == (
        "Miranda requires warnings before custodial interrogation."
    )
    assert normalized_claims[0]["annotation"]["evidence"][1]["evidence_quote"] == (
        "Example narrows the rule."
    )


def test_normalize_claims_for_frontend_handles_unverified_claims():
    claims = [
        {
            "claim_id": "claim-raw",
            "text": "The answer depends on the record.",
            "span": {
                "doc_id": "assistant_response",
                "para_id": 0,
                "sent_id": 1,
                "start_char": 4,
                "end_char": 34,
            },
        }
    ]

    normalized_claims, links = normalize_claims_for_frontend(claims)

    assert links == []
    assert normalized_claims[0]["annotation"]["support_level"] == "unsupported"
    assert normalized_claims[0]["annotation"]["evidence"] == []
