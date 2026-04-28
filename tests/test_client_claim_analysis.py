"""Tests for client-side claim evidence analysis helpers."""

from __future__ import annotations

from src.client.claim_analysis import (
    extract_claim_evaluations,
    find_cases_for_interaction,
    group_evidence_case_references_by_support_level,
)


def test_find_cases_for_interaction_surfaces_top_support_and_contradiction():
    interaction_detail = {
        "interaction": {"id": 12, "conversation_id": 7, "query": "Question", "response": "Answer", "created_at": "2026-04-18T12:00:00+00:00"},
        "claims": [
            {
                "claim_id": "claim-1",
                "text": "Miranda warnings are required.",
                "linked_citations": [
                    {
                        "claim_id": "claim-1",
                        "relationship": "supporting",
                        "score": 0.91,
                        "citation": {
                            "id": "chunk-miranda",
                            "doc_id": "doc-miranda",
                            "case_name": "Miranda v. Arizona",
                            "citation": "384 U.S. 436",
                            "court": "scotus",
                            "court_level": "scotus",
                            "doc_type": "case",
                            "text_preview": "Miranda requires warnings.",
                        },
                    },
                    {
                        "claim_id": "claim-1",
                        "relationship": "contradicting",
                        "score": 0.27,
                        "citation": {
                            "id": "chunk-example",
                            "doc_id": "doc-example",
                            "case_name": "State v. Example",
                            "citation": "123 Example 456",
                            "court": "state",
                            "court_level": "state_supreme",
                            "doc_type": "case",
                            "text_preview": "Example narrows the rule.",
                        },
                    },
                ],
                "annotation": {
                    "support_level": "supported",
                    "explanation": "Retrieved evidence supports the claim.",
                    "response_span": {
                        "doc_id": "assistant_response",
                        "start_char": 0,
                        "end_char": 30,
                        "text": "Miranda warnings are required.",
                    },
                    "evidence": [
                        {
                            "relationship": "supporting",
                            "score": 0.91,
                            "citation": {
                                "id": "chunk-miranda",
                                "doc_id": "doc-miranda",
                                "case_name": "Miranda v. Arizona",
                                "citation": "384 U.S. 436",
                                "court": "scotus",
                                "court_level": "scotus",
                                "doc_type": "case",
                                "text_preview": "Miranda requires warnings.",
                            },
                        }
                    ],
                },
                "verification": {
                    "verdict": "SUPPORTED",
                    "best_supporting_chunk": {
                        "id": "chunk-miranda",
                        "doc_id": "doc-miranda",
                        "case_name": "Miranda v. Arizona",
                        "citation": "384 U.S. 436",
                        "court": "scotus",
                        "court_level": "scotus",
                        "doc_type": "case",
                        "text_preview": "Miranda requires warnings.",
                    },
                    "best_supporting_score": 0.91,
                    "best_contradicting_chunk": {
                        "id": "chunk-example",
                        "doc_id": "doc-example",
                        "case_name": "State v. Example",
                        "citation": "123 Example 456",
                        "court": "state",
                        "court_level": "state_supreme",
                        "doc_type": "case",
                        "text_preview": "Example narrows the rule.",
                    },
                    "best_contradiction_score": 0.27,
                },
            }
        ],
        "citations": [],
        "contradictions": [],
    }

    claims = extract_claim_evaluations(interaction_detail)
    case_analysis = find_cases_for_interaction(interaction_detail)

    assert len(claims) == 1
    assert case_analysis["interaction_id"] == 12
    assert case_analysis["claims"][0]["support_level"] == "supported"
    assert case_analysis["claims"][0]["supporting_case"]["citation"] == "384 U.S. 436"
    assert case_analysis["claims"][0]["contradicting_case"]["citation"] == "123 Example 456"
    assert case_analysis["top_supporting_cases"][0]["score"] == 0.91
    assert case_analysis["top_contradicting_cases"][0]["score"] == 0.27


def test_group_evidence_case_references_by_support_level_dedupes_cases_and_keeps_claims():
    claims = [
        {
            "claim_id": "claim-1",
            "text": "Miranda warnings are required.",
            "annotation": {
                "support_level": "supported",
                "evidence": [
                    {
                        "relationship": "supporting",
                        "score": 0.91,
                        "chunk_id": "chunk-miranda",
                        "doc_id": "doc-miranda",
                        "evidence_quote": "Miranda requires warnings before custodial interrogation.",
                        "source_label": "384 U.S. 436",
                        "citation": {
                            "case_name": "Miranda v. Arizona",
                            "citation": "384 U.S. 436",
                        },
                    }
                ],
            },
        },
        {
            "claim_id": "claim-2",
            "text": "Miranda applies during custodial interrogation.",
            "annotation": {
                "support_level": "supported",
                "evidence": [
                    {
                        "relationship": "supporting",
                        "score": 0.94,
                        "chunk_id": "chunk-miranda",
                        "doc_id": "doc-miranda",
                        "evidence_quote": "Custodial interrogation requires warnings under Miranda.",
                        "source_label": "384 U.S. 436",
                        "citation": {
                            "case_name": "Miranda v. Arizona",
                            "citation": "384 U.S. 436",
                        },
                    }
                ],
            },
        },
        {
            "claim_id": "claim-3",
            "text": "The exception always applies.",
            "annotation": {
                "support_level": "unsupported",
                "evidence": [
                    {
                        "relationship": "contradicting",
                        "score": 0.77,
                        "chunk_id": "chunk-miranda",
                        "doc_id": "doc-miranda",
                        "evidence_quote": "Miranda does not make every exception automatic.",
                        "source_label": "384 U.S. 436",
                        "citation": {
                            "case_name": "Miranda v. Arizona",
                            "citation": "384 U.S. 436",
                        },
                    }
                ],
            },
        },
        {
            "claim_id": "claim-4",
            "text": "No evidence was attached.",
            "annotation": {
                "support_level": "possibly_supported",
                "evidence": [],
            },
        },
    ]

    grouped = group_evidence_case_references_by_support_level(claims)

    assert set(grouped) == {"supported", "possibly_supported", "unsupported"}
    assert len(grouped["supported"]) == 1
    assert grouped["possibly_supported"] == []
    assert len(grouped["unsupported"]) == 1

    supported_ref = grouped["supported"][0]
    assert supported_ref["case_name"] == "Miranda v. Arizona"
    assert supported_ref["reporter_citation"] == "384 U.S. 436"
    assert supported_ref["relationships"] == ["supporting"]
    assert supported_ref["score"] == 0.94
    assert supported_ref["claim_texts"] == [
        "Miranda warnings are required.",
        "Miranda applies during custodial interrogation.",
    ]
    assert supported_ref["evidence_quotes"] == [
        "Miranda requires warnings before custodial interrogation.",
        "Custodial interrogation requires warnings under Miranda.",
    ]

    unsupported_ref = grouped["unsupported"][0]
    assert unsupported_ref["relationships"] == ["contradicting"]
    assert unsupported_ref["claim_texts"] == ["The exception always applies."]
    assert unsupported_ref["evidence_quotes"] == [
        "Miranda does not make every exception automatic."
    ]


def test_find_cases_for_interaction_can_fall_back_to_annotation_evidence():
    interaction_detail = {
        "interaction": {"id": 44},
        "claims": [
            {
                "claim_id": "claim-annotation",
                "text": "The exception is narrow.",
                "annotation": {
                    "support_level": "possibly_supported",
                    "explanation": "Retrieved evidence is directionally supportive but not decisive.",
                    "response_span": {
                        "doc_id": "assistant_response",
                        "start_char": 10,
                        "end_char": 34,
                        "text": "The exception is narrow.",
                    },
                    "evidence": [
                        {
                            "relationship": "supporting",
                            "score": 0.62,
                            "citation": {
                                "id": "chunk-quarles",
                                "doc_id": "doc-quarles",
                                "case_name": "New York v. Quarles",
                                "citation": "467 U.S. 649",
                                "court": "scotus",
                                "court_level": "scotus",
                                "doc_type": "case",
                                "text_preview": "Quarles recognizes a narrow exception.",
                            },
                        }
                    ],
                },
            }
        ],
    }

    case_analysis = find_cases_for_interaction(interaction_detail)

    assert case_analysis["claims"][0]["support_level"] == "possibly_supported"
    assert case_analysis["claims"][0]["supporting_case"]["citation"] == "467 U.S. 649"
