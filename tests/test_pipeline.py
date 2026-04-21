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
    assert meta["verification_chunk_ids"] == ["chunk_scotus"]
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


def test_pipeline_prefers_user_upload_chunks_when_available(
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

    result = pipeline.run(user_id=user["id"], query="Explain Miranda warnings")

    assert result["pipeline"]["user_upload_retrieval_backend_status"] == "ok"
    assert result["pipeline"]["user_upload_retrieval_chunk_count"] == 1
    assert result["pipeline"]["retrieved_chunks"][0]["doc_type"] == "user_upload"
    assert result["pipeline"]["retrieved_chunks"][0]["source_file"] == "motion.txt"
    assert "Source file: motion.txt" in llm.context_calls[0][1][0]


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

    owner_result = pipeline.run(user_id=owner["id"], query="Explain Miranda warnings")
    other_result = pipeline.run(user_id=other["id"], query="Explain Miranda warnings")

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
