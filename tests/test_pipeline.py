"""Tests for end-to-end pipeline behavior."""

from __future__ import annotations

from pathlib import Path
from types import MethodType

import pytest

from src.ingestion.document import LegalChunk
from src.indexing.bm25_index import BM25Index
from src.indexing.index_discovery import IndexArtifacts
from src import pipeline as pipeline_module
from src.pipeline import QueryPipeline
from src.storage.database import Database
from src.verification.nli_verifier import AggregatedScore, NLIVerifier


pytestmark = pytest.mark.smoke


class _FakeLLM:
    def __init__(self):
        self.context_calls = []
        self.direct_queries = []
        self.direct_history_calls = []

    def generate_legal_answer(self, query: str, *, conversation_history=None) -> str:
        self.direct_queries.append(query)
        self.direct_history_calls.append(list(conversation_history or []))
        return "The Court held that Miranda warnings are required."

    def generate_with_context(
        self,
        query: str,
        context,
        max_tokens=None,
        *,
        conversation_history=None,
        case_posture=None,
        response_depth="concise",
    ) -> str:
        _ = max_tokens
        self.context_calls.append((query, list(context), list(conversation_history or []), case_posture, response_depth))
        return "The Court held that Miranda warnings are required."


class _FakeRetriever:
    def retrieve(self, query: str, k: int = 10):
        assert query == "Explain Miranda warnings"
        _ = k
        return [
            LegalChunk(
                id="chunk_scotus",
                doc_id="doc_scotus",
                text="The Supreme Court held that Miranda warnings are required during custodial interrogation.",
                chunk_index=0,
                doc_type="case",
                case_name="Miranda v. Arizona",
                court="scotus",
                court_level="scotus",
                citation="384 U.S. 436",
            ),
            LegalChunk(
                id="chunk_district",
                doc_id="doc_district",
                text="A district court discussed exceptions to Miranda warnings.",
                chunk_index=1,
                doc_type="case",
                case_name="United States v. Example",
                court="dcd",
                court_level="district",
            ),
        ]


def _verifier_with_stubbed_scores() -> NLIVerifier:
    verifier = NLIVerifier(device="cpu")
    score_map = {
        (
            "The Supreme Court held that Miranda warnings are required during custodial interrogation.",
            "The Court held that Miranda warnings are required.",
        ): {"entailment": 0.93, "neutral": 0.05, "contradiction": 0.02},
        (
            "A district court discussed exceptions to Miranda warnings.",
            "The Court held that Miranda warnings are required.",
        ): {"entailment": 0.30, "neutral": 0.55, "contradiction": 0.15},
    }

    def _predict_pairs(self, pairs):
        return [score_map[pair] for pair in pairs]

    verifier._predict_pairs = MethodType(_predict_pairs, verifier)
    return verifier


def test_pipeline_runs_real_verifier_logic_when_retrieval_is_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")
    monkeypatch.setattr(
        pipeline_module,
        "load_user_upload_retriever",
        lambda user_id, *, shared_embedder=None: (None, "unavailable:no_user_upload_index"),
    )

    llm = _FakeLLM()
    pipeline = QueryPipeline(
        db=db,
        llm=llm,
        retriever=_FakeRetriever(),
        verifier=_verifier_with_stubbed_scores(),
    )

    result = pipeline.run(user_id=user["id"], query="Explain Miranda warnings")

    meta = result["pipeline"]
    interaction = result["interaction"]
    user_message = result["user_message"]
    assistant = result["assistant_message"]

    assert meta["retrieval_used"] is True
    assert meta["generation_mode"] == "rag"
    assert meta["retrieval_backend_status"] == "configured"
    assert meta["retrieval_chunk_count"] == 2
    assert meta["verification_backend_status"] == "ok"
    assert meta["claim_count"] == 1
    assert set(meta["timings_ms"]) >= {"retrieval", "generation", "claim_decomposition", "verification"}
    assert meta["claims"][0]["verification"]["is_contradicted"] is False
    assert meta["claims"][0]["annotation"]["support_level"] == "supported"
    assert meta["claims"][0]["annotation"]["response_span"]["text"] == meta["claims"][0]["text"]
    assert [link["relationship"] for link in meta["claims"][0]["linked_citations"]] == [
        "supporting",
        "contradicting",
    ]
    assert meta["claims"][0]["verification"]["best_chunk"]["citation"] == "384 U.S. 436"
    assert meta["claims"][0]["verification"]["best_supporting_chunk"]["citation"] == "384 U.S. 436"
    assert meta["claims"][0]["verification"]["best_supporting_score"] == pytest.approx(0.93)
    assert meta["verification_scope_status"] == "applied:scoped"
    assert meta["verification_scope"]["scope"] == "sentence_evidence"
    assert meta["verification_chunk_ids"] == ["chunk_scotus:sentence_evidence:0"]
    assert meta["claims"][0]["verification"]["best_contradicting_chunk"]["case_name"] == "Miranda v. Arizona"
    assert meta["claims"][0]["verification"]["best_contradiction_score"] == pytest.approx(0.02)
    assert meta["claims"][0]["verification"]["final_score"] > 0.5
    assert meta["conversation_context_message_count"] == 0
    assert meta["claim_support_summary"] == {
        "raw_total": 1,
        "total": 1,
        "supported": 1,
        "possibly_supported": 0,
        "unsupported": 0,
        "excluded_rhetorical": 0,
        "unsupported_ratio": 0.0,
    }
    assert meta["answer_warning"]["show"] is False
    assert meta["answer_warning"]["unsupported_ratio"] == 0.0
    assert meta["answer_blocks"][0]["type"] == "verified_claim"
    assert meta["answer_blocks"][0]["support_level"] == "supported"
    assert meta["answer_block_summary"]["verification_required_blocks"] == 1
    assert interaction["query"] == "Explain Miranda warnings"
    assert user_message["interaction_id"] == interaction["id"]
    assert assistant["role"] == "assistant"
    assert assistant["interaction_id"] == interaction["id"]
    assert llm.direct_queries == []
    assert len(llm.context_calls) == 1
    assert "Case name: Miranda v. Arizona" in llm.context_calls[0][1][0]
    assert "384 U.S. 436" in llm.context_calls[0][1][0]
    assert "Miranda warnings are required" in llm.context_calls[0][1][0]
    assert llm.context_calls[0][2] == []

    with db.connect() as conn:
        claim_rows = conn.execute(
            """
            SELECT claim_text, verdict, confidence, source_citation
            FROM verified_claims
            WHERE interaction_id = ?
            ORDER BY id ASC
            """,
            (interaction["id"],),
        ).fetchall()
        assert len(claim_rows) == 1
        assert claim_rows[0]["source_citation"] == "384 U.S. 436"
        assert claim_rows[0]["verdict"] == meta["claims"][0]["verification"]["verdict"]

        citation_rows = conn.execute(
            """
            SELECT chunk_id, used_in_prompt
            FROM interaction_citations
            WHERE interaction_id = ?
            ORDER BY id ASC
            """,
            (interaction["id"],),
        ).fetchall()
        assert len(citation_rows) == 2
        assert citation_rows[0]["chunk_id"] == "chunk_scotus"
        assert citation_rows[0]["used_in_prompt"] == 1

        claim_link_rows = conn.execute(
            """
            SELECT relationship, score
            FROM claim_citation_links
            WHERE verified_claim_id = (
                SELECT id FROM verified_claims WHERE interaction_id = ? ORDER BY id ASC LIMIT 1
            )
            ORDER BY id ASC
            """,
            (interaction["id"],),
        ).fetchall()
        assert [row["relationship"] for row in claim_link_rows] == ["supporting", "contradicting"]
        assert claim_link_rows[0]["score"] == pytest.approx(0.93)
        assert claim_link_rows[1]["score"] == pytest.approx(0.02)

        conversation_state = conn.execute(
            """
            SELECT summary
            FROM conversation_state
            WHERE conversation_id = ?
            """,
            (result["conversation"]["id"],),
        ).fetchone()
        assert conversation_state is not None
        assert "Explain Miranda warnings" in conversation_state["summary"]


def test_pipeline_research_leads_query_reaches_rag_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")
    monkeypatch.setattr(
        pipeline_module,
        "load_user_upload_retriever",
        lambda user_id, *, shared_embedder=None: (None, "unavailable:no_user_upload_index"),
    )

    class _ResearchLeadRetriever:
        def retrieve(self, query: str, k: int = 10):
            assert query == "Find cases about agency civil penalties and jury trial rights."
            _ = k
            return [
                LegalChunk(
                    id="agency_penalty_chunk",
                    doc_id="agency_penalty_doc",
                    text="The Court discussed civil penalties imposed by an agency.",
                    chunk_index=0,
                    doc_type="case",
                    case_name="Securities and Exchange Commission v. Jarkesy",
                    court_level="scotus",
                ),
                LegalChunk(
                    id="jury_right_chunk",
                    doc_id="jury_right_doc",
                    text="The Court addressed when the Seventh Amendment preserves a jury trial.",
                    chunk_index=0,
                    doc_type="case",
                    case_name="Tull v. United States",
                    court_level="scotus",
                ),
            ]

    llm = _FakeLLM()
    pipeline = QueryPipeline(
        db=db,
        llm=llm,
        retriever=_ResearchLeadRetriever(),
        enable_verification=False,
    )

    result = pipeline.run(
        user_id=user["id"],
        query="Find cases about agency civil penalties and jury trial rights.",
    )

    meta = result["pipeline"]
    assert meta["answer_mode"] == "research_leads"
    assert meta["research_leads_mode"] is True
    assert meta["research_leads_status"] == "applied:research_leads"
    assert meta["pre_generation_refusal_status"] == "not_applied:answerable_or_targeted"
    assert meta["query_grounding"]["query_intent"] == "research_leads"
    assert len(llm.context_calls) == 1
    assert llm.direct_queries == []
    assert llm.context_calls[0][1][0].startswith("Evidence type: research-leads scope constraint")


def test_pipeline_reports_skipped_verification_without_indices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")

    monkeypatch.setattr(
        pipeline_module,
        "_load_default_retriever",
        lambda: (None, "unavailable:no_indices"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_user_upload_retriever",
        lambda user_id, *, shared_embedder=None: (None, "unavailable:no_user_upload_index"),
    )

    llm = _FakeLLM()
    pipeline = QueryPipeline(
        db=db,
        llm=llm,
        retriever=None,
    )

    result = pipeline.run(user_id=user["id"], query="Explain Miranda warnings")

    meta = result["pipeline"]

    assert meta["retrieval_used"] is False
    assert meta["generation_mode"] == "direct"
    assert meta["retrieval_chunk_count"] == 0
    assert meta["verification_backend_status"] == "skipped:no_retriever"
    assert meta["claims"]
    assert meta["claims"][0]["annotation"]["support_level"] == "unsupported"
    assert meta["claim_support_summary"] == {
        "raw_total": 1,
        "total": 1,
        "supported": 0,
        "possibly_supported": 0,
        "unsupported": 1,
        "excluded_rhetorical": 0,
        "unsupported_ratio": 1.0,
    }
    assert meta["answer_warning"]["show"] is True
    assert meta["answer_warning"]["kind"] == "unsupported_majority"
    assert meta["answer_warning"]["unsupported_claim_count"] == 1
    assert llm.direct_queries == ["Explain Miranda warnings"]
    assert llm.direct_history_calls == [[]]
    assert llm.context_calls == []


def test_pipeline_can_skip_decomposition_and_verification_for_generation_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")
    monkeypatch.setattr(
        pipeline_module,
        "load_user_upload_retriever",
        lambda user_id, *, shared_embedder=None: (None, "unavailable:no_user_upload_index"),
    )

    llm = _FakeLLM()
    pipeline = QueryPipeline(
        db=db,
        llm=llm,
        retriever=_FakeRetriever(),
        enable_verification=False,
    )

    result = pipeline.run(user_id=user["id"], query="Explain Miranda warnings")

    meta = result["pipeline"]

    assert meta["retrieval_used"] is True
    assert meta["generation_mode"] == "rag"
    assert meta["verification_enabled"] is False
    assert meta["verification_backend_status"] == "disabled:config"
    assert meta["claim_count"] == 0
    assert meta["claims"] == []
    assert meta["claim_support_summary"] == {
        "raw_total": 0,
        "total": 0,
        "supported": 0,
        "possibly_supported": 0,
        "unsupported": 0,
        "excluded_rhetorical": 0,
        "unsupported_ratio": 0.0,
    }
    assert meta["answer_warning"]["show"] is False
    assert set(meta["timings_ms"]) >= {"retrieval", "generation"}
    assert len(llm.context_calls) == 1
    assert llm.context_calls[0][2] == []


def test_claim_support_summary_excludes_rhetorical_claims_from_warning_ratio():
    summary = pipeline_module._summarize_claim_support(
        [
            {
                "text": "The Court denied certiorari.",
                "annotation": {"support_level": "supported"},
            },
            {
                "text": "This decision is unfortunate according to the dissent.",
                "annotation": {"support_level": "unsupported"},
            },
            {
                "text": "The justice hopes another case will soon be taken up by the Court.",
                "annotation": {"support_level": "unsupported"},
            },
        ]
    )

    assert summary == {
        "raw_total": 3,
        "total": 1,
        "supported": 1,
        "possibly_supported": 0,
        "unsupported": 0,
        "excluded_rhetorical": 2,
        "unsupported_ratio": 0.0,
    }

    warning = pipeline_module._build_answer_warning(summary)
    assert warning["show"] is False


def test_answer_blocks_separate_transition_text_from_verified_claims():
    response = "In short, this is the answer. The Court held that Miranda warnings are required."
    claim_text = "The Court held that Miranda warnings are required."
    start = response.index(claim_text)
    blocks = pipeline_module._build_answer_blocks(
        response,
        [
            {
                "claim_id": "claim-1",
                "text": claim_text,
                "span": {
                    "start_char": start,
                    "end_char": start + len(claim_text),
                },
                "verification": {"verdict": "SUPPORTED"},
                "annotation": {
                    "support_level": "supported",
                    "response_span": {
                        "start_char": start,
                        "end_char": start + len(claim_text),
                        "text": claim_text,
                    },
                    "evidence": [{"relationship": "supporting"}],
                },
            }
        ],
    )

    assert [block["type"] for block in blocks] == ["transition", "verified_claim"]
    assert blocks[0]["verification_required"] is False
    assert blocks[1]["verification_required"] is True
    assert blocks[1]["claim_ids"] == ["claim-1"]


def test_answer_blocks_treat_label_lead_in_as_formatting():
    response = (
        "The holding in Esteras v. United States is: "
        "District courts cannot consider § 3553(a)(2)(A) when revoking supervised release."
    )
    claim_text = "District courts cannot consider § 3553(a)(2)(A) when revoking supervised release."
    start = response.index(claim_text)
    blocks = pipeline_module._build_answer_blocks(
        response,
        [
            {
                "claim_id": "claim-esteras",
                "text": claim_text,
                "span": {
                    "start_char": start,
                    "end_char": start + len(claim_text),
                },
                "verification": {"verdict": "SUPPORTED"},
                "annotation": {
                    "support_level": "supported",
                    "response_span": {
                        "start_char": start,
                        "end_char": start + len(claim_text),
                        "text": claim_text,
                    },
                    "evidence": [{"relationship": "supporting"}],
                },
            }
        ],
    )

    assert [block["type"] for block in blocks] == ["transition", "verified_claim"]
    assert blocks[0]["verification_required"] is False
    assert blocks[0]["support_level"] is None
    assert blocks[1]["support_level"] == "supported"


def test_answer_blocks_promote_missed_legal_assertions_to_unverified_candidates():
    response = "The Court affirmed the judgment."
    blocks = pipeline_module._build_answer_blocks(response, [])

    assert len(blocks) == 1
    assert blocks[0]["type"] == "unverified_claim_candidate"
    assert blocks[0]["verification_required"] is True
    assert blocks[0]["support_level"] == "unsupported"


def test_pipeline_filters_low_value_connective_claims_before_verification():
    raw_claims = pipeline_module.decompose_document(
        {
            "id": "assistant_response",
            "full_text": (
                "So too here. "
                "Insufficient support in retrieved authorities to answer the question. "
                "The Court held that relief was required."
            ),
        }
    )

    filtered, skipped = pipeline_module._filter_claims_for_verification(raw_claims)

    assert [claim.text for claim in filtered] == ["The Court held that relief was required."]
    assert [item["text"] for item in skipped] == [
        "So too here.",
        "Insufficient support in retrieved authorities to answer the question.",
    ]


def test_pipeline_filters_malformed_claims_before_verification():
    raw_claims = pipeline_module.decompose_document(
        {
            "id": "assistant_response",
            "full_text": (
                "Nor does it change just. "
                "This \"did not include agriculture, manufacturing, mining, malum in se crime, or land use. "
                "The majority holds that obtaining a preliminary injunction never entitles a plaintiff "
                "to fees under Section 1988(b),. "
                "The Court held that relief was required."
            ),
        }
    )

    filtered, skipped = pipeline_module._filter_claims_for_verification(raw_claims)

    assert [claim.text for claim in filtered] == ["The Court held that relief was required."]
    assert {item["reason"] for item in skipped} >= {
        "malformed_dangling_clause",
        "malformed_unbalanced_quote",
        "malformed_trailing_punctuation",
    }


def test_pipeline_keeps_lowercase_causal_article_claims_for_verification():
    raw_claims = pipeline_module.decompose_document(
        {
            "id": "assistant_response",
            "full_text": (
                "This defect was not cured because the removal petition did not include "
                "a statement of jurisdiction as required by 28 U.S.C. Section 1441(b)."
            ),
        }
    )

    filtered, skipped = pipeline_module._filter_claims_for_verification(raw_claims)

    assert [claim.text for claim in filtered] == [
        "This defect was not cured.",
        "The removal petition did not include a statement of jurisdiction as required by 28 U.S.C. Section 1441(b).",
    ]
    assert skipped == []


def test_verification_scope_prefers_generation_source_chunks():
    source_chunk = LegalChunk(
        id="source_chunk",
        doc_id="target_doc",
        text="The Court held the rule applies.",
        chunk_index=0,
        doc_type="case",
        case_name="Example v. United States",
        court_level="scotus",
    )
    background_chunk = LegalChunk(
        id="background_chunk",
        doc_id="target_doc",
        text="A party argued the opposite rule.",
        chunk_index=1,
        doc_type="case",
        case_name="Example v. United States",
        court_level="scotus",
    )

    selected, meta = pipeline_module._select_chunks_for_verification(
        retrieved_chunks=[source_chunk, background_chunk],
        prompt_chunks=[source_chunk, background_chunk],
        generation_context_meta={"source_chunk_ids": ["source_chunk"]},
        query_grounding={
            "target_doc_ids": ["target_doc"],
            "target_case": "Example v. United States",
        },
    )

    assert selected == [source_chunk]
    assert meta == {
        "status": "applied:scoped",
        "scope": "generation_source_chunks",
        "tiers": ["generation_source"],
    }


def test_consistency_guard_downgrades_supported_claim_with_missing_case_and_citation():
    class _ClaimPayload:
        text = "Jenkins v. Oregon, 419 U.S. 373 (1975) is the relevant case."

        def to_dict(self):
            return {
                "claim_id": "claim-jenkins",
                "text": self.text,
                "claim_type": "citation",
                "source": "court",
                "certainty": "found",
                "doc_section": "body",
                "span": {
                    "doc_id": "assistant_response",
                    "para_id": 0,
                    "sent_id": 1,
                    "start_char": 0,
                    "end_char": len(self.text),
                },
            }

    burnett_chunk = LegalChunk(
        id="burnett_chunk",
        doc_id="burnett_doc",
        text=(
            "The petition for a writ of certiorari is denied. "
            "JUSTICE GORSUCH, dissenting from the denial of certiorari."
        ),
        chunk_index=0,
        doc_type="case",
        case_name="Burnett v. United States",
        court_level="scotus",
    )
    verdict = AggregatedScore(
        final_score=0.84,
        is_contradicted=False,
        best_chunk_idx=0,
        best_chunk=burnett_chunk,
        support_ratio=1.0,
        component_scores={
            "best_entailment": 0.98,
            "best_contradiction": 0.01,
        },
        best_contradicting_chunk_idx=0,
        best_contradicting_chunk=burnett_chunk,
    )

    payload = pipeline_module._serialize_claim_with_verification(_ClaimPayload(), verdict)

    assert payload["verification"]["verdict"] == "UNSUPPORTED"
    assert payload["verification"]["consistency_guard"]["status"] == "blocked:missing_named_evidence"
    assert payload["verification"]["consistency_guard"]["missing"]["case_names"] == ["Jenkins v. Oregon"]
    assert payload["verification"]["consistency_guard"]["missing"]["citations"] == ["419 U.S. 373"]


def test_consistency_guard_blocks_supported_claim_with_low_target_overlap():
    evidence = LegalChunk(
        id="royal_chunk",
        doc_id="royal_doc",
        text=(
            "When a plaintiff amends her complaint to delete federal-law claims, "
            "the federal court loses supplemental jurisdiction and must remand to state court."
        ),
        chunk_index=0,
        doc_type="case",
        case_name="Royal Canin U. S. A. v. Wullschleger",
        court_level="scotus",
    )

    guard = pipeline_module._claim_evidence_consistency_guard(
        "The Royal Prerogative is a power held by the British monarch.",
        evidence,
        "SUPPORTED",
        query_grounding={"target_case": "Royal Canin U. S. A. v. Wullschleger"},
        generation_context_meta={
            "canonical_answer_fact": (
                "After the plaintiff amended the complaint to remove federal claims, "
                "the federal court lost supplemental jurisdiction."
            )
        },
    )

    assert guard["status"] == "blocked:low_target_evidence_overlap"


def test_high_risk_guard_blocks_unresolved_semantic_false_positive():
    evidence = LegalChunk(
        id="warhol_chunk",
        doc_id="warhol_doc",
        text=(
            "The artist then reproduced the image on a silkscreen and used it "
            "to create a series of portraits."
        ),
        chunk_index=0,
        doc_type="case",
        case_name="Andy Warhol Foundation for Visual Arts, Inc. v. Goldsmith",
        court_level="scotus",
        citation="598 U.S. 508",
    )

    guard = pipeline_module._claim_evidence_consistency_guard(
        (
            "The Government's decision to fulfill its end of a bargain despite "
            "actual knowledge of widespread DBE program violations is strong "
            "evidence that those requirements are not material."
        ),
        evidence,
        "SUPPORTED",
        query_grounding={"status": "not_resolved:no_signal", "source": "unresolved"},
        generation_context_meta={"status": "applied:missing_case_topic_fallback"},
    )

    assert guard["status"] == "blocked:high_risk_low_source_alignment"


def test_high_risk_guard_blocks_related_authority_fallback_claim():
    evidence = LegalChunk(
        id="tyler_chunk",
        doc_id="tyler_doc",
        text=(
            "Minnesota allowed the State to sell property to satisfy a tax debt, "
            "and the taxpayer alleged that retaining the surplus violated the Takings Clause."
        ),
        chunk_index=0,
        doc_type="case",
        case_name="Tyler v. Hennepin County",
        court_level="scotus",
        citation="598 U.S. 631",
    )

    guard = pipeline_module._claim_evidence_consistency_guard(
        (
            "Based on related retrieved authorities, the Minnesota statute requiring "
            "any surplus to revert to the owner is unconstitutional because it "
            "extinguishes property rights in violation of the Takings Clause."
        ),
        evidence,
        "POSSIBLE_SUPPORT",
        query_grounding={
            "status": "not_resolved:missing_explicit_case_topic_fallback",
            "source": "explicit_case",
            "explicit_case": "Galactic Mining Co. v. Mars Colony Authority",
            "target_case": None,
        },
        generation_context_meta={"status": "applied:missing_case_topic_fallback"},
    )

    assert guard["status"] == "blocked:high_risk_related_authority_claim"


def test_high_risk_guard_blocks_unresolved_citation_mismatch():
    evidence = LegalChunk(
        id="taamneh_chunk",
        doc_id="taamneh_doc",
        text="The statute does not apply unless the defendant knowingly provided substantial assistance.",
        chunk_index=0,
        doc_type="case",
        case_name="Twitter, Inc. v. Taamneh",
        court_level="scotus",
        citation="598 U.S. 471",
    )

    guard = pipeline_module._claim_evidence_consistency_guard(
        "The statute does not apply to payments or gifts to officials unless they are made corruptly.",
        evidence,
        "SUPPORTED",
        query_grounding={
            "status": "not_resolved:citation_unresolved",
            "source": "citation_unresolved",
            "target_citation": "999 U.S. 999",
        },
        generation_context_meta={"status": "not_applied:no_target_case"},
    )

    assert guard["status"] == "blocked:high_risk_unresolved_citation_mismatch"


def test_high_risk_guard_does_not_low_overlap_block_research_leads():
    evidence = LegalChunk(
        id="tariff_chunk",
        doc_id="tariff_doc",
        text="The President's argument rests on statutory authorization under IEEPA and separation-of-powers principles.",
        chunk_index=0,
        doc_type="case",
        case_name="Example Tariff Case",
        court_level="scotus",
    )

    guard = pipeline_module._claim_evidence_consistency_guard(
        'The broad meaning of the term "regulate" includes traditional means such as quotas, embargoes, and tariffs.',
        evidence,
        "POSSIBLE_SUPPORT",
        query_grounding={
            "status": "not_resolved:llm_route:research_leads",
            "source": "llm_route:research_leads",
            "query_intent": "research_leads",
        },
        generation_context_meta={"status": "applied:research_leads"},
    )

    assert guard["status"] == "passed"


def test_high_risk_guard_allows_unresolved_claim_with_direct_sentence_support():
    evidence = LegalChunk(
        id="cedar_chunk",
        doc_id="cedar_doc",
        text="A party invoking force majeure must give written notice within fourteen calendar days after the event begins.",
        chunk_index=0,
        doc_type="user_upload",
        source_file="cedar_supply_agreement.txt",
    )

    guard = pipeline_module._claim_evidence_consistency_guard(
        "A party invoking force majeure must give written notice within fourteen calendar days after the event begins.",
        evidence,
        "SUPPORTED",
        query_grounding={"status": "not_resolved:user_upload", "user_upload_mentioned": True},
        generation_context_meta={"status": "applied:user_upload_context"},
    )

    assert guard["status"] == "passed"


def test_high_risk_guard_does_not_apply_to_resolved_target_case():
    evidence = LegalChunk(
        id="loper_chunk",
        doc_id="loper_doc",
        text="The deference that Chevron requires of courts reviewing agency action cannot be squared with the APA.",
        chunk_index=0,
        doc_type="case",
        case_name="Loper Bright Enterprises v. Raimondo",
        court_level="scotus",
        citation="603 U.S. 369",
    )

    guard = pipeline_module._claim_evidence_consistency_guard(
        "The deference that Chevron requires of courts reviewing agency action cannot be squared with the APA.",
        evidence,
        "SUPPORTED",
        query_grounding={
            "status": "resolved:explicit_case",
            "source": "explicit_case",
            "target_case": "Loper Bright Enterprises v. Raimondo",
        },
        generation_context_meta={},
    )

    assert guard["status"] == "passed"


def test_legal_polarity_guard_blocks_chevron_controls_after_loper_bright():
    evidence = LegalChunk(
        id="loper_chunk",
        doc_id="loper_doc",
        text="The deference that Chevron requires of courts reviewing agency action cannot be squared with the APA.",
        chunk_index=0,
        doc_type="case",
        case_name="Loper Bright Enterprises v. Raimondo",
        court_level="scotus",
        citation="603 U.S. 369",
    )

    guard = pipeline_module._claim_evidence_consistency_guard(
        "The rule now controlling agency interpretations is Chevron deference.",
        evidence,
        "SUPPORTED",
        query_grounding={
            "status": "resolved:explicit_case",
            "source": "explicit_case",
            "target_case": "Loper Bright Enterprises v. Raimondo",
        },
        generation_context_meta={},
    )

    assert guard["status"] == "blocked:legal_polarity_conflict"


def test_research_leads_guard_demotes_weak_source_alignment():
    evidence = LegalChunk(
        id="standing_chunk",
        doc_id="standing_doc",
        text=(
            "The causation requirement rules out attenuated links where government action "
            "is far removed from predictable ripple effects."
        ),
        chunk_index=0,
        doc_type="case",
        case_name="FDA v. Alliance for Hippocratic Medicine",
        court_level="scotus",
    )

    guard = pipeline_module._claim_evidence_consistency_guard(
        "The fuel producers have Article III standing to challenge EPA's approval of California regulations.",
        evidence,
        "SUPPORTED",
        query_grounding={
            "status": "not_resolved:llm_route:research_leads",
            "source": "llm_route:research_leads",
            "query_intent": "research_leads",
        },
        generation_context_meta={"status": "applied:research_leads"},
    )

    assert guard["status"] == "demote:research_leads_weak_source_alignment"


def test_research_leads_guard_keeps_strong_source_alignment():
    evidence = LegalChunk(
        id="fuel_chunk",
        doc_id="fuel_doc",
        text=(
            "Fuel producers have Article III standing to challenge EPA approval of "
            "California regulations because reduced gasoline demand is a concrete injury."
        ),
        chunk_index=0,
        doc_type="case",
        case_name="Diamond Alternative Energy, LLC v. EPA",
        court_level="scotus",
    )

    guard = pipeline_module._claim_evidence_consistency_guard(
        "The fuel producers have Article III standing to challenge EPA's approval of California regulations.",
        evidence,
        "SUPPORTED",
        query_grounding={
            "status": "not_resolved:llm_route:research_leads",
            "source": "llm_route:research_leads",
            "query_intent": "research_leads",
        },
        generation_context_meta={"status": "applied:research_leads"},
    )

    assert guard["status"] == "passed"


def test_query_subject_guard_blocks_source_discipline_incidental_match():
    evidence = LegalChunk(
        id="sackett_chunk",
        doc_id="sackett_doc",
        text="This did not include agriculture, manufacturing, mining, malum in se crime, or land use.",
        chunk_index=0,
        doc_type="case",
        case_name="Sackett v. EPA",
        court_level="scotus",
        citation="598 U.S. 651",
    )

    guard = pipeline_module._claim_evidence_consistency_guard(
        "This did not include agriculture, manufacturing, mining, malum in se crime, or land use.",
        evidence,
        "SUPPORTED",
        query_grounding={
            "query": (
                "Use only retrieved sources. If the retrieved sources do not mention "
                "a holding about moon mining, say that instead. What is the moon mining holding?"
            ),
            "status": "resolved:retrieval_convergence",
            "source": "retrieval_convergence",
            "target_case": "Sackett v. EPA",
        },
        generation_context_meta={"status": "applied:sentence_evidence"},
    )

    assert guard["status"] == "blocked:query_subject_mismatch"
    assert guard["missing_subject_tokens"] == ["moon"]


def test_query_subject_guard_allows_source_discipline_subject_match():
    evidence = LegalChunk(
        id="moon_chunk",
        doc_id="moon_doc",
        text="The retrieved source states that moon mining rights are not addressed by the holding.",
        chunk_index=0,
        doc_type="case",
        case_name="Example Space Mining Case",
        court_level="scotus",
    )

    guard = pipeline_module._claim_evidence_consistency_guard(
        "The retrieved source states that moon mining rights are not addressed by the holding.",
        evidence,
        "SUPPORTED",
        query_grounding={
            "query": (
                "Use only retrieved sources. If the retrieved sources do not mention "
                "a holding about moon mining, say that instead. What is the moon mining holding?"
            ),
            "status": "not_resolved:no_signal",
            "source": "unresolved",
        },
        generation_context_meta={"status": "not_applied:no_target_case"},
    )

    assert guard["status"] == "passed"


def test_pipeline_filters_prompt_chunks_to_target_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")
    monkeypatch.setattr(
        pipeline_module,
        "load_user_upload_retriever",
        lambda user_id, *, shared_embedder=None: (None, "unavailable:no_user_upload_index"),
    )

    class _CaseRetriever:
        def retrieve(self, query: str, k: int = 10):
            assert query == "In Burnett v. United States, what did the Supreme Court decide about supervised release?"
            _ = k
            return [
                LegalChunk(
                    id="burnett_chunk_1",
                    doc_id="burnett_doc",
                    text="Burnett discusses supervised release and a Sixth Amendment issue.",
                    chunk_index=0,
                    doc_type="case",
                    case_name="Burnett v. United States",
                    court="scotus",
                    court_level="scotus",
                ),
                LegalChunk(
                    id="burnett_chunk_2",
                    doc_id="burnett_doc",
                    text="The petition for certiorari was denied in Burnett.",
                    chunk_index=1,
                    doc_type="case",
                    case_name=None,
                    court="scotus",
                    court_level="scotus",
                ),
                LegalChunk(
                    id="other_chunk",
                    doc_id="other_doc",
                    text="Learning Resources discusses the major questions doctrine.",
                    chunk_index=0,
                    doc_type="case",
                    case_name="Learning Resources, Inc. v. Trump",
                    court="scotus",
                    court_level="scotus",
                ),
            ]

    llm = _FakeLLM()
    pipeline = QueryPipeline(
        db=db,
        llm=llm,
        retriever=_CaseRetriever(),
        enable_verification=False,
    )

    result = pipeline.run(
        user_id=user["id"],
        query="In Burnett v. United States, what did the Supreme Court decide about supervised release?",
    )

    meta = result["pipeline"]

    assert meta["retrieval_chunk_count"] == 3
    assert meta["prompt_chunk_count"] == 2
    assert meta["prompt_case_filter_status"] == "applied"
    assert meta["target_case_name"] == "Burnett v. United States"
    assert meta["prompt_chunk_ids"] == ["burnett_chunk_1", "burnett_chunk_2"]
    assert len(llm.context_calls) == 1
    prompt_context = llm.context_calls[0][1]
    assert len(prompt_context) == 2
    assert all("Burnett" in item or "burnett_doc" in item for item in prompt_context)
    assert all("Learning Resources" not in item for item in prompt_context)

    with db.connect() as conn:
        citation_rows = conn.execute(
            """
            SELECT chunk_id, used_in_prompt
            FROM interaction_citations
            WHERE interaction_id = ?
            ORDER BY id ASC
            """,
            (result["interaction"]["id"],),
        ).fetchall()
        assert [row["chunk_id"] for row in citation_rows] == [
            "burnett_chunk_1",
            "burnett_chunk_2",
            "other_chunk",
        ]
        assert [row["used_in_prompt"] for row in citation_rows] == [1, 1, 0]


def test_pipeline_missing_explicit_case_refusal_is_not_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")
    monkeypatch.setattr(
        pipeline_module,
        "load_user_upload_retriever",
        lambda user_id, *, shared_embedder=None: (None, "unavailable:no_user_upload_index"),
    )

    class _UnrelatedRetriever:
        def retrieve(self, query: str, k: int = 10):
            _ = query, k
            return [
                LegalChunk(
                    id="unrelated_chunk",
                    doc_id="unrelated_doc",
                    text="Held: The judgment of the court of appeals is reversed.",
                    chunk_index=0,
                    doc_type="case",
                    case_name="Unrelated v. United States",
                    court="scotus",
                    court_level="scotus",
                )
            ]

    llm = _FakeLLM()
    pipeline = QueryPipeline(
        db=db,
        llm=llm,
        retriever=_UnrelatedRetriever(),
        enable_verification=False,
    )

    result = pipeline.run(
        user_id=user["id"],
        query="What did Missing v. Unknown hold?",
    )

    meta = result["pipeline"]
    assert result["assistant_message"]["content"] == (
        "Insufficient support in retrieved authorities to answer the question."
    )
    assert meta["answer_mode"] == "refusal"
    assert meta["pre_generation_refusal_status"] == "applied:explicit_target_not_retrieved"
    assert meta["response_override_status"] == "not_applied:pre_generation_refusal"
    assert meta["cert_denial_guard_status"] == "not_applied:pre_generation_refusal"
    assert llm.context_calls == []
    assert llm.direct_queries == []


def test_pipeline_missing_case_can_answer_related_topic_with_disclaimer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")
    monkeypatch.setattr(
        pipeline_module,
        "load_user_upload_retriever",
        lambda user_id, *, shared_embedder=None: (None, "unavailable:no_user_upload_index"),
    )

    class _TopicLLM(_FakeLLM):
        def generate_with_context(
            self,
            query: str,
            context,
            max_tokens=None,
            *,
            conversation_history=None,
            case_posture=None,
            response_depth="concise",
        ) -> str:
            self.context_calls.append((query, list(context), list(conversation_history or []), case_posture, response_depth))
            return "Adverse possession requires possession that satisfies the elements stated in the retrieved authority."

    class _TopicFallbackRetriever:
        def __init__(self):
            self.queries: list[str] = []

        def retrieve(self, query: str, k: int = 10):
            self.queries.append(query)
            _ = k
            if "adverse possession" not in query.lower():
                return [
                    LegalChunk(
                        id="unrelated_chunk",
                        doc_id="unrelated_doc",
                        text="An unrelated criminal procedure case discusses harmless error.",
                        chunk_index=0,
                        doc_type="case",
                        case_name="Other v. United States",
                        court="scotus",
                        court_level="scotus",
                    )
                ]
            return [
                LegalChunk(
                    id="property_chunk",
                    doc_id="property_doc",
                    text=(
                        "Property law recognizes adverse possession when possession is actual, "
                        "open and notorious, exclusive, hostile, and continuous for the statutory period."
                    ),
                    chunk_index=0,
                    doc_type="case",
                    case_name="Property Owner v. Possessor",
                    court="state",
                    court_level="state_supreme",
                )
            ]

    retriever = _TopicFallbackRetriever()
    llm = _TopicLLM()
    pipeline = QueryPipeline(
        db=db,
        llm=llm,
        retriever=retriever,
        enable_verification=False,
    )

    result = pipeline.run(
        user_id=user["id"],
        query="In Missing v. Unknown, what does property law say about adverse possession?",
    )

    response = result["assistant_message"]["content"]
    meta = result["pipeline"]
    assert response.startswith(
        "I do not have Missing v. Unknown in the retrieved database, so I cannot say what that case held."
    )
    assert "Based on related retrieved authorities" in response
    assert "adverse possession requires possession" in response.lower()
    assert meta["answer_mode"] == "missing_case_topic_fallback"
    assert meta["topic_fallback_used"] is True
    assert meta["topic_fallback_status"] == "applied"
    assert meta["missing_target_case"] == "Missing v. Unknown"
    assert meta["target_case_answered"] is False
    assert meta["prompt_case_filter_status"] == "applied:missing_case_topic_fallback"
    assert meta["prompt_chunk_ids"] == ["property_chunk"]
    assert len(retriever.queries) == 2
    assert "Missing v. Unknown" not in retriever.queries[1]
    assert len(llm.context_calls) == 1
    assert "Missing target case: Missing v. Unknown" in llm.context_calls[0][1][0]
    assert "Property law recognizes adverse possession" in llm.context_calls[0][1][1]


def test_pipeline_metadata_target_retrieval_rescues_explicit_case_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")
    monkeypatch.setattr(
        pipeline_module,
        "load_user_upload_retriever",
        lambda user_id, *, shared_embedder=None: (None, "unavailable:no_user_upload_index"),
    )

    burnett_chunks = [
        LegalChunk(
            id="burnett_chunk_0",
            doc_id="burnett_doc",
            text=(
                "The petition for a writ of certiorari is denied. "
                "JUSTICE GORSUCH, dissenting from the denial of certiorari."
            ),
            chunk_index=0,
            doc_type="case",
            case_name="Burnett v. United States",
            court="scotus",
            court_level="scotus",
        ),
        LegalChunk(
            id="burnett_chunk_1",
            doc_id="burnett_doc",
            text="Justice Gorsuch would have taken the case to consider the Sixth Amendment argument.",
            chunk_index=1,
            doc_type="case",
            case_name="Burnett v. United States",
            court="scotus",
            court_level="scotus",
        ),
    ]

    class _RetrieverWithMetadata:
        bm25_index = BM25Index(burnett_chunks)

        def retrieve(self, query: str, k: int = 10):
            _ = query, k
            return [
                LegalChunk(
                    id="other_chunk",
                    doc_id="other_doc",
                    text="An unrelated case discusses the Eighth Amendment death penalty framework.",
                    chunk_index=0,
                    doc_type="case",
                    case_name="Other v. United States",
                    court="scotus",
                    court_level="scotus",
                )
            ]

    llm = _FakeLLM()
    pipeline = QueryPipeline(
        db=db,
        llm=llm,
        retriever=_RetrieverWithMetadata(),
        enable_verification=False,
    )

    result = pipeline.run(
        user_id=user["id"],
        query=(
            "In Burnett v. United States, give a bottom line and then explain the posture "
            "in two short paragraphs."
        ),
    )

    meta = result["pipeline"]
    assert meta["target_metadata_retrieval_status"] == "applied"
    assert meta["prompt_case_filter_status"] == "applied"
    assert meta["target_case_name"] == "Burnett v. United States"
    assert meta["prompt_chunk_ids"] == ["burnett_chunk_0", "burnett_chunk_1"]
    assert all("death penalty" not in item.lower() for item in llm.context_calls[0][1])


def test_pipeline_extracts_case_posture_and_passes_it_to_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")
    monkeypatch.setattr(
        pipeline_module,
        "load_user_upload_retriever",
        lambda user_id, *, shared_embedder=None: (None, "unavailable:no_user_upload_index"),
    )

    class _PostureRetriever:
        def retrieve(self, query: str, k: int = 10):
            assert query == "In Burnett v. United States, what did the Supreme Court decide about supervised release?"
            _ = k
            return [
                LegalChunk(
                    id="burnett_chunk_1",
                    doc_id="burnett_doc",
                    text=(
                        "The petition for a writ of certiorari is denied. "
                        "JUSTICE GORSUCH, dissenting from the denial of certiorari. "
                        "I would have taken this case to consider the Sixth Amendment issue."
                    ),
                    chunk_index=0,
                    doc_type="case",
                    case_name="Burnett v. United States",
                    court="scotus",
                    court_level="scotus",
                ),
                LegalChunk(
                    id="burnett_chunk_2",
                    doc_id="burnett_doc",
                    text="Mr. Burnett argued that any sentence above the statutory maximum required jury findings.",
                    chunk_index=1,
                    doc_type="case",
                    case_name="Burnett v. United States",
                    court="scotus",
                    court_level="scotus",
                ),
            ]

    llm = _FakeLLM()
    pipeline = QueryPipeline(
        db=db,
        llm=llm,
        retriever=_PostureRetriever(),
        enable_verification=False,
    )

    result = pipeline.run(
        user_id=user["id"],
        query="In Burnett v. United States, what did the Supreme Court decide about supervised release?",
    )

    meta = result["pipeline"]
    assert meta["case_posture_status"] == "inferred"
    assert meta["case_posture"] == {
        "target_case": "Burnett v. United States",
        "decision_type": "cert_denial",
        "court_action": "denied certiorari",
        "opinion_role": "dissent_from_denial",
        "author": "Gorsuch",
        "is_separate_opinion": True,
        "source_chunk_ids": ["burnett_chunk_1", "burnett_chunk_2"],
    }
    assert len(llm.context_calls) == 1
    assert llm.context_calls[0][3] == meta["case_posture"]


def test_pipeline_overrides_cert_denial_response_with_named_author(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")
    monkeypatch.setattr(
        pipeline_module,
        "load_user_upload_retriever",
        lambda user_id, *, shared_embedder=None: (None, "unavailable:no_user_upload_index"),
    )

    class _PostureRetriever:
        def retrieve(self, query: str, k: int = 10):
            assert query == "In Burnett v. United States, what did the Supreme Court do in the case?"
            _ = k
            return [
                LegalChunk(
                    id="burnett_chunk_1",
                    doc_id="burnett_doc",
                    text=(
                        "The petition for a writ of certiorari is denied. "
                        "JUSTICE GORSUCH, dissenting from the denial of certiorari."
                    ),
                    chunk_index=0,
                    doc_type="case",
                    case_name="Burnett v. United States",
                    court="scotus",
                    court_level="scotus",
                )
            ]

    class _GarbageLLM(_FakeLLM):
        def generate_with_context(
            self,
            query: str,
            context,
            max_tokens=None,
            *,
            conversation_history=None,
            case_posture=None,
            response_depth="concise",
        ) -> str:
            _ = max_tokens
            self.context_calls.append((query, list(context), list(conversation_history or []), case_posture, response_depth))
            return "The Court denied certiorari.\n\nJustice [name] dissented."

    llm = _GarbageLLM()
    pipeline = QueryPipeline(
        db=db,
        llm=llm,
        retriever=_PostureRetriever(),
        enable_verification=False,
    )

    result = pipeline.run(
        user_id=user["id"],
        query="In Burnett v. United States, what did the Supreme Court do in the case?",
    )

    assert result["assistant_message"]["content"] == (
        "The Supreme Court denied certiorari in Burnett v. United States.\n\n"
        "Justice Gorsuch dissented from the denial of certiorari."
    )
    assert result["pipeline"]["response_override_status"] == "applied:cert_denial_posture"
    assert result["pipeline"]["response_override_meta"]["author"] == "Gorsuch"


def test_pipeline_grounds_vague_followup_from_conversation_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")
    monkeypatch.setattr(
        pipeline_module,
        "load_user_upload_retriever",
        lambda user_id, *, shared_embedder=None: (None, "unavailable:no_user_upload_index"),
    )

    class _FollowupRetriever:
        def __init__(self):
            self.queries = []

        def retrieve(self, query: str, k: int = 10):
            self.queries.append(query)
            _ = k
            return [
                LegalChunk(
                    id="burnett_chunk_1",
                    doc_id="burnett_doc",
                    text=(
                        "The petition for a writ of certiorari is denied. "
                        "JUSTICE GORSUCH, dissenting from the denial of certiorari."
                    ),
                    chunk_index=0,
                    doc_type="case",
                    case_name="Burnett v. United States",
                    court="scotus",
                    court_level="scotus",
                )
            ]

    retriever = _FollowupRetriever()
    llm = _FakeLLM()
    pipeline = QueryPipeline(
        db=db,
        llm=llm,
        retriever=retriever,
        enable_verification=False,
    )

    first = pipeline.run(
        user_id=user["id"],
        query="In Burnett v. United States, what did the Supreme Court do in the case?",
    )
    second = pipeline.run(
        user_id=user["id"],
        conversation_id=first["conversation"]["id"],
        query="Who wrote the separate opinion?",
    )

    assert retriever.queries == [
        "In Burnett v. United States, what did the Supreme Court do in the case?",
        "In Burnett v. United States, who wrote the dissent from the denial of certiorari?",
    ]
    assert second["pipeline"]["followup_grounding_status"] == "applied:conversation_state"
    assert second["pipeline"]["effective_query"] == (
        "In Burnett v. United States, who wrote the dissent from the denial of certiorari?"
    )
    assert second["pipeline"]["target_case_name"] == "Burnett v. United States"
    assert second["assistant_message"]["content"] == (
        "The Supreme Court denied certiorari in Burnett v. United States.\n\n"
        "Justice Gorsuch dissented from the denial of certiorari."
    )


def test_pipeline_does_not_replace_cert_denial_issue_analysis_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")
    monkeypatch.setattr(
        pipeline_module,
        "load_user_upload_retriever",
        lambda user_id, *, shared_embedder=None: (None, "unavailable:no_user_upload_index"),
    )

    class _BurnettRetriever:
        def retrieve(self, query: str, k: int = 10):
            assert query == (
                "What did Burnett say about whether supervised release can push prison time "
                "past the statutory maximum without a jury?"
            )
            _ = k
            return [
                LegalChunk(
                    id="burnett_chunk_1",
                    doc_id="burnett_doc",
                    text=(
                        "The petition for a writ of certiorari is denied. "
                        "JUSTICE GORSUCH, dissenting from the denial of certiorari. "
                        "Mr. Burnett argued that supervised release punishment above the statutory maximum "
                        "required jury findings under the Sixth Amendment."
                    ),
                    chunk_index=0,
                    doc_type="case",
                    case_name="Burnett v. United States",
                    court="scotus",
                    court_level="scotus",
                ),
                LegalChunk(
                    id="esteras_chunk_1",
                    doc_id="esteras_doc",
                    text="Esteras concerns supervised release revocation factors.",
                    chunk_index=0,
                    doc_type="case",
                    case_name="Esteras v. United States",
                    court="scotus",
                    court_level="scotus",
                ),
            ]

    class _IssueLLM(_FakeLLM):
        def generate_with_context(
            self,
            query: str,
            context,
            max_tokens=None,
            *,
            conversation_history=None,
            case_posture=None,
            response_depth="concise",
        ) -> str:
            _ = max_tokens
            self.context_calls.append((query, list(context), list(conversation_history or []), case_posture, response_depth))
            return (
                "The Supreme Court denied certiorari in Burnett v. United States.\n\n"
                "Justice Gorsuch's dissent said Mr. Burnett raised a Sixth Amendment issue about "
                "supervised release punishment above the statutory maximum without jury findings."
            )

    llm = _IssueLLM()
    pipeline = QueryPipeline(
        db=db,
        llm=llm,
        retriever=_BurnettRetriever(),
        enable_verification=False,
    )

    result = pipeline.run(
        user_id=user["id"],
        query=(
            "What did Burnett say about whether supervised release can push prison time "
            "past the statutory maximum without a jury?"
        ),
    )

    assert result["assistant_message"]["content"] == (
        "The Supreme Court denied certiorari in Burnett v. United States.\n\n"
        "Justice Gorsuch's dissent said Mr. Burnett raised a Sixth Amendment issue about "
        "supervised release punishment above the statutory maximum without jury findings."
    )
    assert result["pipeline"]["query_grounding_status"] == "resolved:short_name"
    assert result["pipeline"]["query_grounding"]["query_intent"] == "issue_analysis"
    assert result["pipeline"]["response_override_status"] == "not_applied:intent_not_posture_or_author"


def test_pipeline_preserves_detailed_response_instead_of_canonical_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")
    monkeypatch.setattr(
        pipeline_module,
        "load_user_upload_retriever",
        lambda user_id, *, shared_embedder=None: (None, "unavailable:no_user_upload_index"),
    )
    query = (
        "In Esteras v. United States, explain why retribution matters to "
        "supervised-release revocation and include bullets."
    )

    class _EsterasRetriever:
        def retrieve(self, query: str, k: int = 10):
            _ = query, k
            return [
                LegalChunk(
                    id="esteras_chunk_1",
                    doc_id="esteras_doc",
                    text=(
                        "District courts cannot consider section 3553(a)(2)(A) when revoking supervised release. "
                        "Section 3553(a)(2)(A) concerns retribution. "
                        "The Court explained that Congress omitted that factor from the revocation statute."
                    ),
                    chunk_index=0,
                    doc_type="case",
                    case_name="Esteras v. United States",
                    court="scotus",
                    court_level="scotus",
                )
            ]

    class _DetailedLLM(_FakeLLM):
        def generate_with_context(
            self,
            query: str,
            context,
            max_tokens=None,
            *,
            conversation_history=None,
            case_posture=None,
            response_depth="concise",
        ) -> str:
            _ = max_tokens
            self.context_calls.append((query, list(context), list(conversation_history or []), case_posture, response_depth))
            return (
                "District courts cannot consider section 3553(a)(2)(A) when revoking supervised release.\n\n"
                "- Section 3553(a)(2)(A) concerns retribution.\n"
                "- Congress omitted that factor from the revocation statute."
            )

    llm = _DetailedLLM()
    pipeline = QueryPipeline(
        db=db,
        llm=llm,
        retriever=_EsterasRetriever(),
        enable_verification=False,
    )

    result = pipeline.run(user_id=user["id"], query=query)

    assert result["assistant_message"]["content"].startswith(
        "District courts cannot consider section 3553(a)(2)(A)"
    )
    assert "- Section 3553(a)(2)(A) concerns retribution." in result["assistant_message"]["content"]
    assert result["pipeline"]["response_depth"] == "detailed"
    assert result["pipeline"]["response_override_status"] == "not_applied:detailed_response_depth"
    assert result["pipeline"]["generation_context_count"] >= 3
    assert llm.context_calls[0][4] == "detailed"


def test_pipeline_limits_named_case_prompt_chunks_to_small_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")
    monkeypatch.setattr(
        pipeline_module,
        "load_user_upload_retriever",
        lambda user_id, *, shared_embedder=None: (None, "unavailable:no_user_upload_index"),
    )

    class _ManyChunkRetriever:
        def retrieve(self, query: str, k: int = 10):
            assert query == "In Esteras v. United States, what did the Supreme Court decide about supervised release?"
            _ = k
            return [
                LegalChunk(
                    id=f"esteras_chunk_{idx}",
                    doc_id="esteras_doc",
                    text=f"Esteras chunk {idx} text.",
                    chunk_index=idx,
                    doc_type="case",
                    case_name="Esteras v. United States",
                    court="scotus",
                    court_level="scotus",
                )
                for idx in range(6)
            ]

    llm = _FakeLLM()
    pipeline = QueryPipeline(
        db=db,
        llm=llm,
        retriever=_ManyChunkRetriever(),
        enable_verification=False,
    )

    result = pipeline.run(
        user_id=user["id"],
        query="In Esteras v. United States, what did the Supreme Court decide about supervised release?",
    )

    meta = result["pipeline"]
    assert meta["retrieval_chunk_count"] == 6
    assert meta["prompt_chunk_count"] == 6
    assert meta["target_case_prompt_candidate_count"] == 6
    assert meta["target_case_prompt_limit"] == 8
    assert meta["prompt_chunk_ids"] == [
        "esteras_chunk_0",
        "esteras_chunk_1",
        "esteras_chunk_2",
        "esteras_chunk_3",
        "esteras_chunk_4",
        "esteras_chunk_5",
    ]
    assert len(llm.context_calls) == 1
    assert len(llm.context_calls[0][1]) == 6


def test_pipeline_reuses_lazy_verifier_instance_across_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")
    monkeypatch.setattr(
        pipeline_module,
        "load_user_upload_retriever",
        lambda user_id, *, shared_embedder=None: (None, "unavailable:no_user_upload_index"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_load_default_retriever",
        lambda: (None, "unavailable:no_indices"),
    )
    monkeypatch.setattr(pipeline_module.VERIFICATION, "verifier_mode", "live")

    created_verifiers: list[object] = []

    class _CountingVerifier:
        def __init__(self) -> None:
            created_verifiers.append(self)

        def verify_claims_batch(self, claims, chunks):
            _ = chunks
            return [NLIVerifier._empty_result() for _ in claims]

    monkeypatch.setattr(pipeline_module, "NLIVerifier", _CountingVerifier)

    llm = _FakeLLM()
    pipeline = QueryPipeline(
        db=db,
        llm=llm,
        retriever=_FakeRetriever(),
    )

    pipeline.run(user_id=user["id"], query="Explain Miranda warnings")
    pipeline.run(user_id=user["id"], query="Explain Miranda warnings")

    assert len(created_verifiers) == 1
    assert pipeline.verifier is created_verifiers[0]


def test_pipeline_uses_recent_messages_as_conversation_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")
    conversation = db.create_conversation(user["id"], "Miranda follow-up")
    monkeypatch.setattr(
        pipeline_module,
        "load_user_upload_retriever",
        lambda user_id, *, shared_embedder=None: (None, "unavailable:no_user_upload_index"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_load_default_retriever",
        lambda: (None, "unavailable:no_indices"),
    )

    db.add_message(conversation["id"], "user", "Explain Miranda warnings.")
    db.add_message(conversation["id"], "assistant", "Miranda requires warnings during custodial interrogation.")
    db.add_message(conversation["id"], "user", "What about the public safety exception?")

    llm = _FakeLLM()
    pipeline = QueryPipeline(db=db, llm=llm, retriever=None)

    result = pipeline.run(
        user_id=user["id"],
        query="Does the public safety exception change that rule?",
        conversation_id=conversation["id"],
    )

    assert result["pipeline"]["conversation_context_message_count"] == 2
    assert llm.direct_queries == ["Does the public safety exception change that rule?"]
    assert llm.direct_history_calls == [[
        {
            "role": "assistant",
            "content": "Miranda requires warnings during custodial interrogation.",
        },
        {"role": "user", "content": "What about the public safety exception?"},
    ]]


def test_pipeline_preloads_embedder_and_verifier_models(monkeypatch: pytest.MonkeyPatch):
    loaded: list[str] = []
    monkeypatch.setattr(pipeline_module.VERIFICATION, "verifier_mode", "live")

    class _PreloadEmbedder:
        def _load_model(self):
            loaded.append("embedder")
            return object()

    class _PreloadVerifier:
        def _load_model(self):
            loaded.append("verifier")
            return object(), object(), object()

    retriever = _FakeRetriever()
    retriever.embedder = _PreloadEmbedder()

    pipeline = QueryPipeline(
        db=Database(":memory:"),
        llm=_FakeLLM(),
        retriever=retriever,
        verifier=_PreloadVerifier(),
    )

    status = pipeline.preload_models()

    assert status == {
        "retrieval_embedder": "ok",
        "verification_model": "ok",
    }
    assert loaded == ["embedder", "verifier"]


def test_pipeline_uses_heuristic_verifier_when_configured(monkeypatch, tmp_path: Path):
    db = Database(tmp_path / "heuristic.db")
    db.initialize()
    user = db.create_user("heuristic", "hashed")
    monkeypatch.setattr(
        pipeline_module,
        "load_user_upload_retriever",
        lambda user_id, *, shared_embedder=None: (None, "unavailable:no_user_upload_index"),
    )
    monkeypatch.setattr(pipeline_module.VERIFICATION, "verifier_mode", "heuristic")

    created = []

    class _CountingHeuristicVerifier:
        def __init__(self):
            created.append(self)

        def verify_claims_batch(self, claims, chunks):
            _ = chunks
            return [NLIVerifier._empty_result() for _ in claims]

    monkeypatch.setattr(pipeline_module, "HeuristicNLIVerifier", _CountingHeuristicVerifier)

    llm = _FakeLLM()
    pipeline = QueryPipeline(
        db=db,
        llm=llm,
        retriever=_FakeRetriever(),
    )

    result = pipeline.run(user_id=user["id"], query="Explain Miranda warnings")

    assert len(created) == 1
    assert pipeline.verifier is created[0]
    assert result["pipeline"]["verification_verifier_mode"] == "heuristic"


def test_pipeline_preload_skips_live_model_for_heuristic_mode(monkeypatch):
    loaded: list[str] = []

    class _PreloadEmbedder:
        def _load_model(self):
            loaded.append("embedder")
            return object()

    class _FailingVerifier:
        def _load_model(self):
            loaded.append("verifier")
            raise AssertionError("live verifier should not preload in heuristic mode")

    retriever = _FakeRetriever()
    retriever.embedder = _PreloadEmbedder()
    monkeypatch.setattr(pipeline_module.VERIFICATION, "verifier_mode", "heuristic")

    pipeline = QueryPipeline(
        db=Database(":memory:"),
        llm=_FakeLLM(),
        retriever=retriever,
        verifier=_FailingVerifier(),
    )

    status = pipeline.preload_models()

    assert status == {
        "retrieval_embedder": "ok",
        "verification_model": "configured:heuristic",
    }
    assert loaded == ["embedder"]


def test_pipeline_verification_falls_back_to_heuristic(monkeypatch, tmp_path):
    from src import pipeline as pipeline_module

    db = Database(tmp_path / "fallback.db")
    db.initialize()
    user = db.create_user("fallback@example.com", "hash")

    class _FixedLLM:
        host = "http://localhost:11434"
        model = "llama3.1:8b"

        def generate_with_context(
            self,
            query,
            context,
            conversation_history=None,
            case_posture=None,
            response_depth="concise",
        ):
            _ = query, context, conversation_history, case_posture, response_depth
            return "The court denied relief."

    class _StaticRetriever:
        def retrieve(self, query):
            _ = query
            return [
                LegalChunk(
                    id="chunk-1",
                    doc_id="doc-1",
                    text="The court denied relief.",
                    chunk_index=0,
                    doc_type="case",
                    court_level="scotus",
                )
            ]

    class _ExplodingVerifier:
        def verify_claims_batch(self, claims, chunks):
            _ = claims, chunks
            raise OSError("paging file too small")

    class _FallbackVerifier:
        def verify_claims_batch(self, claims, chunks):
            _ = claims, chunks
            return [NLIVerifier._empty_result() for _ in claims]

    monkeypatch.setattr(pipeline_module.VERIFICATION, "fallback_to_heuristic_on_error", True)
    monkeypatch.setattr(pipeline_module, "HeuristicNLIVerifier", _FallbackVerifier)

    pipeline = QueryPipeline(
        db=db,
        llm=_FixedLLM(),
        retriever=_StaticRetriever(),
        verifier=_ExplodingVerifier(),
        enable_verification=True,
    )

    result = pipeline.run(user_id=user["id"], query="What happened?")

    assert result["pipeline"]["verification_backend_status"] == "warning:fallback:HeuristicNLIVerifier"
    assert result["pipeline"]["verification_error"] == {
        "type": "OSError",
        "message": "paging file too small",
    }
    assert result["pipeline"]["verification_fallback"] == {
        "verifier": "HeuristicNLIVerifier",
    }


def test_load_default_retriever_uses_discovered_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_discover():
        return IndexArtifacts(
            bm25_path=tmp_path / "nli_100_bm25.pkl",
            chroma_path=tmp_path / "nli_100_chroma",
            collection_name="nli_100_chunks",
            summary_path=tmp_path / "nli_100_index_summary.json",
            source="summary:nli_100_index_summary.json",
        )

    def fake_load_bm25_index(index_path: Path):
        captured["bm25_path"] = index_path
        return object()

    def fake_load_vector_store(chroma_path: Path, *, collection_name: str | None = None):
        captured["chroma_path"] = chroma_path
        captured["collection_name"] = collection_name
        return object(), object()

    class _FakeHybridRetriever:
        def __init__(self, **kwargs):
            captured["retriever_kwargs"] = kwargs

    monkeypatch.setattr(pipeline_module, "discover_index_artifacts", fake_discover)
    monkeypatch.setattr(pipeline_module, "_load_bm25_index", fake_load_bm25_index)
    monkeypatch.setattr(pipeline_module, "_load_vector_store", fake_load_vector_store)
    monkeypatch.setattr(pipeline_module, "HybridRetriever", _FakeHybridRetriever)

    retriever, status = pipeline_module._load_default_retriever()

    assert status == "ok"
    assert isinstance(retriever, _FakeHybridRetriever)
    assert captured["bm25_path"] == tmp_path / "nli_100_bm25.pkl"
    assert captured["chroma_path"] == tmp_path / "nli_100_chroma"
    assert captured["collection_name"] == "nli_100_chunks"


def test_pipeline_skips_user_upload_chunks_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")

    def unexpected_load_user_upload_retriever(user_id: int, *, shared_embedder=None):
        _ = user_id, shared_embedder
        raise AssertionError("User upload retriever should not load unless requested.")

    monkeypatch.setattr(
        pipeline_module,
        "load_user_upload_retriever",
        unexpected_load_user_upload_retriever,
    )

    llm = _FakeLLM()
    pipeline = QueryPipeline(
        db=db,
        llm=llm,
        retriever=_FakeRetriever(),
        enable_verification=False,
    )

    result = pipeline.run(user_id=user["id"], query="Explain Miranda warnings")

    assert result["pipeline"]["include_uploaded_chunks"] is False
    assert result["pipeline"]["user_upload_retrieval_backend_status"] == "disabled:not_requested"
    assert result["pipeline"]["user_upload_retrieval_chunk_count"] == 0
    assert all(chunk["doc_type"] != "user_upload" for chunk in result["pipeline"]["retrieved_chunks"])


def test_pipeline_prefers_user_upload_chunks_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")

    upload_chunk = LegalChunk(
        id="upload_chunk_1",
        doc_id="upload_doc_1",
        text="The uploaded motion states that the arbitration clause survives termination.",
        chunk_index=0,
        doc_type="user_upload",
        source_file="motion.txt",
    )

    class _UploadRetriever:
        def retrieve(self, query: str, k: int = 10):
            assert query == "Explain Miranda warnings"
            _ = k
            return [upload_chunk]

    monkeypatch.setattr(
        pipeline_module,
        "load_user_upload_retriever",
        lambda user_id, *, shared_embedder=None: (_UploadRetriever(), "ok"),
    )

    llm = _FakeLLM()
    pipeline = QueryPipeline(
        db=db,
        llm=llm,
        retriever=_FakeRetriever(),
        enable_verification=False,
    )

    result = pipeline.run(
        user_id=user["id"],
        query="Explain Miranda warnings",
        include_uploaded_chunks=True,
    )

    assert result["pipeline"]["include_uploaded_chunks"] is True
    assert result["pipeline"]["user_upload_retrieval_backend_status"] == "ok"
    assert result["pipeline"]["user_upload_retrieval_chunk_count"] == 1
    assert result["pipeline"]["retrieved_chunks"][0]["doc_type"] == "user_upload"
    assert result["pipeline"]["retrieved_chunks"][0]["source_file"] == "motion.txt"
    assert "Source file: motion.txt" in llm.context_calls[0][1][0]


def test_pipeline_rewrites_public_retrieval_query_for_upload_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")

    query = (
        "Compare my uploaded Northstar Model H-17 draft to relevant corpus cases or precedent. "
        "How does Dr. Lena Marquez's causation methodology compare to prior cases?"
    )
    upload_chunk = LegalChunk(
        id="upload_motion:0",
        doc_id="upload_motion",
        text=(
            "Riley v. Northstar Home Robotics, Inc. Dr. Lena Marquez failed to connect her "
            "observations to a reliable causation methodology and failed to account for "
            "alternative ignition sources."
        ),
        chunk_index=0,
        doc_type="user_upload",
        source_file="northstar_upload_probe.txt",
    )
    public_chunk = LegalChunk(
        id="case_expert_reliability:0",
        doc_id="case_expert_reliability",
        text=(
            "Expert testimony may be excluded when a causation opinion lacks reliable "
            "methodology and fails to address alternative causes."
        ),
        chunk_index=0,
        doc_type="case",
        case_name="Example Expert Reliability Case",
    )

    class _UploadRetriever:
        def retrieve(self, received_query: str, k: int = 10):
            assert received_query == query
            _ = k
            return [upload_chunk]

    class _RecordingPublicRetriever:
        def __init__(self):
            self.queries = []

        def retrieve(self, received_query: str, k: int = 10):
            _ = k
            self.queries.append(received_query)
            assert "reliable causation methodology" in received_query
            assert "alternative causes" in received_query
            assert "Northstar" not in received_query
            assert "Marquez" not in received_query
            return [public_chunk]

    monkeypatch.setattr(
        pipeline_module,
        "load_user_upload_retriever",
        lambda user_id, *, shared_embedder=None: (_UploadRetriever(), "ok"),
    )

    retriever = _RecordingPublicRetriever()
    llm = _FakeLLM()
    pipeline = QueryPipeline(
        db=db,
        llm=llm,
        retriever=retriever,
        enable_verification=False,
    )

    result = pipeline.run(user_id=user["id"], query=query, include_uploaded_chunks=True)
    meta = result["pipeline"]

    assert len(retriever.queries) == 1
    assert retriever.queries[0] == meta["public_retrieval_query"]
    assert meta["public_retrieval_query"] != query
    assert meta["public_retrieval_query_meta"]["status"] == "applied:user_upload_comparison_rewrite"
    assert meta["prompt_case_filter_status"] == "applied:user_upload_with_comparison_authorities"
    assert meta["prompt_chunk_ids"] == ["upload_motion:0", "case_expert_reliability:0"]


def test_pipeline_keeps_public_chunks_for_upload_comparison_without_strict_rerank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")

    query = (
        "Compare my uploaded Northstar Model H-17 draft to relevant corpus cases or precedent. "
        "How does Dr. Lena Marquez's causation methodology compare to prior cases?"
    )
    upload_chunk = LegalChunk(
        id="upload_motion:0",
        doc_id="upload_motion",
        text=(
            "Riley v. Northstar Home Robotics, Inc. Dr. Lena Marquez failed to connect her "
            "observations to a reliable causation methodology and failed to account for "
            "alternative ignition sources."
        ),
        chunk_index=0,
        doc_type="user_upload",
        source_file="northstar_upload_probe.txt",
    )
    weak_public_chunk = LegalChunk(
        id="case_weak:0",
        doc_id="case_weak",
        text="The court described a theory of liability as substantial and not frivolous.",
        chunk_index=0,
        doc_type="case",
        case_name="Unrelated Injunction Case",
    )

    class _UploadRetriever:
        def retrieve(self, received_query: str, k: int = 10):
            assert received_query == query
            _ = k
            return [upload_chunk]

    class _WeakPublicRetriever:
        def retrieve(self, received_query: str, k: int = 10):
            _ = received_query, k
            return [weak_public_chunk]

    monkeypatch.setattr(
        pipeline_module,
        "load_user_upload_retriever",
        lambda user_id, *, shared_embedder=None: (_UploadRetriever(), "ok"),
    )

    llm = _FakeLLM()
    pipeline = QueryPipeline(
        db=db,
        llm=llm,
        retriever=_WeakPublicRetriever(),
        enable_verification=False,
    )

    result = pipeline.run(user_id=user["id"], query=query, include_uploaded_chunks=True)
    meta = result["pipeline"]

    assert meta["public_rerank_meta"]["status"] == "not_applied:disabled_query_variant_only"
    assert meta["public_retrieval_chunk_count"] == 1
    assert meta["prompt_case_filter_status"] == "applied:user_upload_with_comparison_authorities"
    assert meta["prompt_chunk_ids"] == ["upload_motion:0", "case_weak:0"]
    assert meta["retrieved_chunks"][0]["doc_type"] == "user_upload"
    assert any(chunk["id"] == "case_weak:0" for chunk in meta["retrieved_chunks"])


def test_pipeline_verifies_generated_claims_against_user_upload_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")

    upload_chunk = LegalChunk(
        id="upload_motion:0",
        doc_id="upload_motion",
        text=(
            "The uploaded motion states that the arbitration clause survives termination. "
            "The uploaded motion does not say that the court granted summary judgment."
        ),
        chunk_index=0,
        doc_type="user_upload",
        source_file="motion.txt",
    )

    class _UploadRetriever:
        def retrieve(self, query: str, k: int = 10):
            assert query == "Using my uploaded motion, what does it say?"
            _ = k
            return [upload_chunk]

    class _NoPublicRetriever:
        def retrieve(self, query: str, k: int = 10):
            assert query == "Using my uploaded motion, what does it say?"
            _ = k
            return []

    class _UploadAnswerLLM:
        def __init__(self):
            self.context_calls = []

        def generate_with_context(
            self,
            query: str,
            context,
            max_tokens=None,
            *,
            conversation_history=None,
            case_posture=None,
            response_depth="concise",
        ) -> str:
            _ = max_tokens, conversation_history, case_posture, response_depth
            self.context_calls.append((query, list(context)))
            return (
                "The uploaded motion states that the arbitration clause survives termination. "
                "The uploaded motion states that the court granted summary judgment."
            )

        def generate_legal_answer(self, query: str, *, conversation_history=None) -> str:
            raise AssertionError("Expected retrieval-grounded generation for user upload.")

    class _UploadVerifier:
        def __init__(self):
            self.calls = []

        def verify_claims_batch(self, claims, chunks):
            self.calls.append(([claim.text for claim in claims], [chunk.id for chunk in chunks]))
            assert chunks
            assert all(chunk.doc_type == "user_upload" for chunk in chunks)
            assert all(
                getattr(chunk, "verification_source_chunk_id", None) == "upload_motion:0"
                for chunk in chunks
            )
            verdicts = []
            for claim in claims:
                if "survives termination" in claim.text:
                    verdicts.append(
                        AggregatedScore(
                            final_score=0.80,
                            is_contradicted=False,
                            best_chunk_idx=0,
                            best_chunk=upload_chunk,
                            support_ratio=1.0,
                            component_scores={
                                "best_entailment": 0.90,
                                "best_contradiction": 0.02,
                            },
                            best_contradicting_chunk_idx=0,
                            best_contradicting_chunk=upload_chunk,
                        )
                    )
                elif "granted summary judgment" in claim.text:
                    verdicts.append(
                        AggregatedScore(
                            final_score=0.10,
                            is_contradicted=False,
                            best_chunk_idx=0,
                            best_chunk=upload_chunk,
                            support_ratio=0.0,
                            component_scores={
                                "best_entailment": 0.10,
                                "best_contradiction": 0.05,
                            },
                            best_contradicting_chunk_idx=0,
                            best_contradicting_chunk=upload_chunk,
                        )
                    )
                else:
                    raise AssertionError(f"Unexpected claim text: {claim.text}")
            return verdicts

    monkeypatch.setattr(
        pipeline_module,
        "load_user_upload_retriever",
        lambda user_id, *, shared_embedder=None: (_UploadRetriever(), "ok"),
    )

    llm = _UploadAnswerLLM()
    verifier = _UploadVerifier()
    pipeline = QueryPipeline(
        db=db,
        llm=llm,
        retriever=_NoPublicRetriever(),
        verifier=verifier,
        enable_verification=True,
    )

    result = pipeline.run(
        user_id=user["id"],
        query="Using my uploaded motion, what does it say?",
        include_uploaded_chunks=True,
    )
    meta = result["pipeline"]

    assert meta["verification_backend_status"] == "ok"
    assert meta["prompt_case_filter_status"] == "applied:user_upload_only"
    assert meta["verification_scope_status"] == "applied:scoped"
    assert meta["verification_scope"]["tiers"] == ["sentence_evidence"]
    assert all(chunk_id.startswith("upload_motion:0:sentence_evidence:") for chunk_id in meta["verification_chunk_ids"])
    assert meta["claim_support_summary"] == {
        "raw_total": 1,
        "total": 1,
        "supported": 1,
        "possibly_supported": 0,
        "unsupported": 0,
        "excluded_rhetorical": 0,
        "unsupported_ratio": 0.0,
    }
    assert len(verifier.calls) == 2
    initial_claims, initial_chunk_ids = verifier.calls[0]
    assert all(chunk_id.startswith("upload_motion:0:sentence_evidence:") for chunk_id in initial_chunk_ids)
    assert any("survives termination" in claim for claim in initial_claims)
    assert any("granted summary judgment" in claim for claim in initial_claims)
    repaired_claims, repaired_chunk_ids = verifier.calls[1]
    assert all(chunk_id.startswith("upload_motion:0:sentence_evidence:") for chunk_id in repaired_chunk_ids)
    assert all("granted summary judgment" not in claim for claim in repaired_claims)
    assert "survives termination" in result["assistant_message"]["content"]
    assert "granted summary judgment" not in result["assistant_message"]["content"]
    assert meta["response_repair_status"] == "applied:unsupported_claim_repair"
    assert meta["response_repair_meta"]["summary"] == {
        "raw_total": 2,
        "total": 2,
        "supported": 1,
        "possibly_supported": 0,
        "unsupported": 1,
        "excluded_rhetorical": 0,
        "unsupported_ratio": 0.5,
    }


def test_pipeline_scopes_user_upload_retrieval_to_current_user(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    owner = db.create_user("owner", "hashed")
    other = db.create_user("other", "hashed")
    calls: list[int] = []

    upload_chunk = LegalChunk(
        id="upload_chunk_owner",
        doc_id="upload_doc_owner",
        text="Owner-only uploaded evidence.",
        chunk_index=0,
        doc_type="user_upload",
        source_file="owner.txt",
    )

    class _OwnerUploadRetriever:
        def retrieve(self, query: str, k: int = 10):
            assert query == "Explain Miranda warnings"
            _ = k
            return [upload_chunk]

    def fake_load_user_upload_retriever(user_id: int, *, shared_embedder=None):
        _ = shared_embedder
        calls.append(user_id)
        if user_id == owner["id"]:
            return _OwnerUploadRetriever(), "ok"
        return None, "unavailable:no_user_upload_index"

    monkeypatch.setattr(
        pipeline_module,
        "load_user_upload_retriever",
        fake_load_user_upload_retriever,
    )
    monkeypatch.setattr(
        pipeline_module,
        "_load_default_retriever",
        lambda: (None, "unavailable:no_indices"),
    )

    llm = _FakeLLM()
    pipeline = QueryPipeline(
        db=db,
        llm=llm,
        retriever=None,
        enable_verification=False,
    )

    owner_result = pipeline.run(
        user_id=owner["id"],
        query="Explain Miranda warnings",
        include_uploaded_chunks=True,
    )
    other_result = pipeline.run(
        user_id=other["id"],
        query="Explain Miranda warnings",
        include_uploaded_chunks=True,
    )

    assert owner_result["pipeline"]["user_upload_retrieval_chunk_count"] == 1
    assert owner_result["pipeline"]["retrieval_used"] is True
    assert owner_result["pipeline"]["generation_mode"] == "rag"
    assert other_result["pipeline"]["user_upload_retrieval_chunk_count"] == 0
    assert other_result["pipeline"]["retrieval_used"] is False
    assert calls == [owner["id"], other["id"]]


def test_pipeline_surfaces_llm_runtime_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")

    monkeypatch.setattr(
        pipeline_module,
        "_load_default_retriever",
        lambda: (None, "unavailable:no_indices"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_user_upload_retriever",
        lambda user_id, *, shared_embedder=None: (None, "unavailable:no_user_upload_index"),
    )

    class _FailingLLM:
        host = "http://localhost:11434"
        model = "llama3.1:8b"

        def generate_legal_answer(self, query: str, *, conversation_history=None) -> str:
            _ = (query, conversation_history)
            raise RuntimeError("unable to allocate CUDA0 buffer")

    pipeline = QueryPipeline(
        db=db,
        llm=_FailingLLM(),
        retriever=None,
    )

    result = pipeline.run(user_id=user["id"], query="Explain Miranda warnings")

    assert result["assistant_message"]["content"].startswith(
        "The configured Ollama provider could not generate a response."
    )
    assert "unable to allocate CUDA0 buffer" in result["assistant_message"]["content"]
    assert result["pipeline"]["llm_backend_status"] == "error:RuntimeError"
    assert result["pipeline"]["llm_config"] == {
        "host": "http://localhost:11434",
        "model": "llama3.1:8b",
    }
    assert result["pipeline"]["llm_error"] == {
        "type": "RuntimeError",
        "message": "unable to allocate CUDA0 buffer",
        "host": "http://localhost:11434",
        "model": "llama3.1:8b",
    }
    assert result["pipeline"]["verification_backend_status"] == "skipped:llm_error"
    assert result["pipeline"]["claim_count"] == 0
    assert result["pipeline"]["claims"] == []
    assert result["pipeline"]["claim_citation_links"] == []


# =============================================================================
# RAG Targeting and Evidence Quality Tests (Phase 1-4)
# =============================================================================


def test_pipeline_refuses_unresolved_explicit_citation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that fake citations like 999 U.S. 999 are refused without LLM call."""
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")
    monkeypatch.setattr(
        pipeline_module,
        "load_user_upload_retriever",
        lambda user_id, *, shared_embedder=None: (None, "unavailable:no_user_upload_index"),
    )

    # Retriever returns unrelated chunks (simulating what happens with 999 U.S. 999)
    class _UnrelatedRetriever:
        def retrieve(self, query: str, k: int = 10):
            _ = query, k
            return [
                LegalChunk(
                    id="unrelated_chunk",
                    doc_id="unrelated_doc",
                    text="Some unrelated case about corporate liability.",
                    chunk_index=0,
                    doc_type="case",
                    case_name="Acme Corp v. Smith",
                    court="scotus",
                    court_level="scotus",
                    citation="555 U.S. 123",
                )
            ]

    llm = _FakeLLM()
    pipeline = QueryPipeline(
        db=db,
        llm=llm,
        retriever=_UnrelatedRetriever(),
        enable_verification=False,
    )

    result = pipeline.run(
        user_id=user["id"],
        query="Summarize 999 U.S. 999 and explain its rule.",
    )

    # Should refuse without calling LLM
    assert result["assistant_message"]["content"] == (
        "I could not find 999 U.S. 999 in the retrieved database, so I cannot summarize its rule."
    )
    assert result["pipeline"]["answer_mode"] == "refusal"
    assert result["pipeline"]["pre_generation_refusal_status"] == "applied:explicit_citation_not_retrieved"
    assert result["pipeline"]["query_grounding_status"] == "not_resolved:citation_unresolved"
    # No LLM context calls should be made
    assert llm.context_calls == []
    assert llm.direct_queries == []


def test_pipeline_resolves_known_citation_with_normalized_spacing():
    """Test that citation variants like 606 U.S. 185, 606 U. S. 185, 606 US 185 all resolve."""
    # Create chunks with different citation formats
    esteras_chunks = [
        LegalChunk(
            id="esteras_chunk_0",
            doc_id="esteras_doc",
            text="District courts cannot consider section 3553(a)(2)(A) when revoking supervised release.",
            chunk_index=0,
            doc_type="case",
            case_name="Esteras v. United States",
            citation="606 U.S. 185",  # Standard format
            court_level="scotus",
        ),
    ]

    citation_variants = [
        "606 U.S. 185",
        "606 U. S. 185",  # Extra space
        "606 US 185",     # No periods
    ]

    for variant in citation_variants:
        grounding = pipeline_module._resolve_query_grounding(
            f"What did {variant} hold about supervised release?",
            esteras_chunks,
        )
        assert grounding["status"] == "resolved:citation"
        assert grounding["target_case"] == "Esteras v. United States"
        assert grounding["target_citation"] == variant


def test_burnett_sentence_evidence_filters_fragments():
    """Test that fragmented sentences like 'is the right to...' are filtered."""
    # Chunks with known bad fragments from Burnett corpus
    burnett_chunks_with_fragments = [
        LegalChunk(
            id="burnett_chunk_0",
            doc_id="burnett_doc",
            text=(
                "The petition for a writ of certiorari is denied. "
                "JUSTICE GORSUCH, dissenting from the denial of certiorari."
            ),
            chunk_index=0,
            doc_type="case",
            case_name="Burnett v. United States",
            court_level="scotus",
        ),
        LegalChunk(
            id="burnett_chunk_bad",
            doc_id="burnett_doc",
            text=(
                "is the right to have a jury decide any contested facts "
                "under the reasonable doubt standard. Should the government "
                "seek prison time beyond that because of his latest alleged "
                "supervised release violations, Mr. Burnett submitted, the "
                "Sixth Amendment required the government to prove its case."
            ),
            chunk_index=1,
            doc_type="case",
            case_name="Burnett v. United States",
            court_level="scotus",
        ),
    ]

    query = "In Burnett v. United States, what did the Supreme Court decide about supervised release?"
    query_tokens = pipeline_module._content_tokens(query)

    # Score sentences from both chunks
    scores = []
    for chunk in burnett_chunks_with_fragments:
        for idx, sentence in enumerate(pipeline_module._split_into_sentences(chunk.text)):
            score = pipeline_module._score_named_case_evidence_sentence(
                query_tokens=query_tokens,
                sentence=sentence,
                chunk=chunk,
                target_case="Burnett v. United States",
                sentence_index=idx,
            )
            scores.append((sentence[:60], score))

    # Bad fragments should score 0.0
    bad_fragments = [
        "is the right to have a jury decide any contested facts",
        "Should the government seek prison time beyond that",
    ]
    
    for sentence_preview, score in scores:
        for bad_fragment in bad_fragments:
            if bad_fragment.lower() in sentence_preview.lower():
                assert score == 0.0, f"Bad fragment should score 0: {sentence_preview}"

    # Good sentences should have positive scores
    good_sentences = [s for s, score in scores if score > 0]
    assert len(good_sentences) > 0, "Should have some valid sentences"


def test_canonical_answer_fact_for_burnett_cert_denial():
    """Test that Burnett cert denial generates correct canonical fact."""
    evidence_sentences = [
        {"sentence": "The petition for a writ of certiorari is denied."},
        {"sentence": "JUSTICE GORSUCH, dissenting from the denial of certiorari."},
        {"sentence": "is the right to have a jury decide any contested facts"},  # Bad fragment
    ]

    query = "In Burnett v. United States, what did the Supreme Court decide about supervised release?"
    
    canonical = pipeline_module._canonical_answer_fact(
        query,
        evidence_sentences,
        explicit_holding=None,
    )

    # Should return canonical fact about cert denial and dissent
    assert canonical is not None
    assert "denied certiorari" in canonical.lower()
    assert "burnett" in canonical.lower()
    assert "gorsuch" in canonical.lower()
    assert "dissent" in canonical.lower()
    # Should NOT say "the Court held" or similar
    assert "court held" not in canonical.lower()
    assert "court held that" not in canonical.lower()


def test_burnett_posture_override_applies_to_posture_queries():
    """Test that posture override applies to posture/author queries, not holding queries."""
    # Posture override only applies to posture/author queries per existing logic
    posture = {
        "target_case": "Burnett v. United States",
        "decision_type": "cert_denial",
        "court_action": "denied certiorari",
        "opinion_role": "dissent_from_denial",
        "author": "Gorsuch",
        "is_separate_opinion": True,
    }

    # For posture query - should apply override
    response_posture, meta_posture = pipeline_module._apply_case_posture_response_override(
        query="In Burnett v. United States, what did the Court do?",
        response="The Court decided the case.",
        case_posture=posture,
        query_intent="posture",
    )
    assert meta_posture["status"] == "applied:cert_denial_posture"
    assert "denied certiorari" in response_posture.lower()
    assert "gorsuch" in response_posture.lower()

    # For author query - should apply override
    response_author, meta_author = pipeline_module._apply_case_posture_response_override(
        query="In Burnett v. United States, who wrote the dissent?",
        response="Justice Gorsuch wrote about the Sixth Amendment.",
        case_posture=posture,
        query_intent="author",
    )
    assert meta_author["status"] == "applied:cert_denial_posture"

    # For holding query - should NOT apply posture override (canonical fact handles this)
    response_holding, meta_holding = pipeline_module._apply_case_posture_response_override(
        query="What did Burnett hold about supervised release?",
        response="The Court held something.",
        case_posture=posture,
        query_intent="holding",
    )
    assert meta_holding["status"] == "not_applied:intent_not_posture_or_author"
