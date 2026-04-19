"""Tests for end-to-end pipeline behavior."""

from __future__ import annotations

from pathlib import Path
from types import MethodType

import pytest

from src.ingestion.document import LegalChunk
from src.indexing.index_discovery import IndexArtifacts
from src import pipeline as pipeline_module
from src.pipeline import QueryPipeline
from src.storage.database import Database
from src.verification.nli_verifier import NLIVerifier


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
    ) -> str:
        _ = max_tokens
        self.context_calls.append((query, list(context), list(conversation_history or [])))
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
    assert meta["claims"][0]["verification"]["is_contradicted"] is False
    assert meta["claims"][0]["verification"]["best_chunk"]["citation"] == "384 U.S. 436"
    assert meta["claims"][0]["verification"]["best_supporting_chunk"]["citation"] == "384 U.S. 436"
    assert meta["claims"][0]["verification"]["best_supporting_score"] == pytest.approx(0.93)
    assert meta["claims"][0]["verification"]["best_contradicting_chunk"]["case_name"] == "United States v. Example"
    assert meta["claims"][0]["verification"]["best_contradiction_score"] == pytest.approx(0.15)
    assert meta["claims"][0]["verification"]["final_score"] > 0.5
    assert meta["conversation_context_message_count"] == 0
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
        assert claim_link_rows[1]["score"] == pytest.approx(0.15)

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
    assert len(llm.context_calls) == 1
    assert llm.context_calls[0][2] == []


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

    assert result["pipeline"]["conversation_context_message_count"] == 3
    assert llm.direct_queries == ["Does the public safety exception change that rule?"]
    assert llm.direct_history_calls == [[
        {"role": "user", "content": "Explain Miranda warnings."},
        {
            "role": "assistant",
            "content": "Miranda requires warnings during custodial interrogation.",
        },
        {"role": "user", "content": "What about the public safety exception?"},
    ]]


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
