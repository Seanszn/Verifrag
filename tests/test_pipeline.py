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


def test_pipeline_runs_real_verifier_logic_when_retrieval_is_available(tmp_path: Path):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")

    llm = _FakeLLM()
    pipeline = QueryPipeline(
        db=db,
        llm=llm,
        retriever=_FakeRetriever(),
        verifier=_verifier_with_stubbed_scores(),
    )

    result = pipeline.run(user_id=user["id"], query="Explain Miranda warnings")

    meta = result["pipeline"]
    assistant = result["assistant_message"]

    assert meta["retrieval_used"] is True
    assert meta["generation_mode"] == "rag"
    assert meta["retrieval_backend_status"] == "configured"
    assert meta["retrieval_chunk_count"] == 2
    assert meta["verification_backend_status"] == "ok"
    assert meta["claim_count"] == 1
    assert meta["claims"][0]["verification"]["is_contradicted"] is False
    assert meta["claims"][0]["verification"]["best_chunk"]["citation"] == "384 U.S. 436"
    assert meta["claims"][0]["verification"]["final_score"] > 0.5
    assert meta["conversation_context_message_count"] == 0
    assert assistant["role"] == "assistant"
    assert llm.direct_queries == []
    assert len(llm.context_calls) == 1
    assert "Case name: Miranda v. Arizona" in llm.context_calls[0][1][0]
    assert "384 U.S. 436" in llm.context_calls[0][1][0]
    assert "Miranda warnings are required" in llm.context_calls[0][1][0]
    assert llm.context_calls[0][2] == []


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


def test_pipeline_can_skip_decomposition_and_verification_for_generation_only(tmp_path: Path):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")

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


def test_pipeline_uses_recent_messages_as_conversation_context(tmp_path: Path):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")
    conversation = db.create_conversation(user["id"], "Miranda follow-up")

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
