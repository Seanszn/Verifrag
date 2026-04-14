"""Tests for end-to-end pipeline behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from src import pipeline as pipeline_module
from src.ingestion.document import LegalChunk
from src.pipeline import QueryPipeline
from src.storage.database import Database
from src.verification.nli_verifier import AggregatedScore


pytestmark = pytest.mark.smoke


class _FakeLLM:
    def __init__(self):
        self.context_calls = []
        self.direct_queries = []

    def generate_legal_answer(self, query: str) -> str:
        self.direct_queries.append(query)
        return "Miranda warnings are required during custodial interrogation."

    def generate_with_context(self, query: str, context, max_tokens=None) -> str:
        _ = max_tokens
        self.context_calls.append((query, list(context)))
        return "Miranda warnings are required during custodial interrogation."


class _FakeRetriever:
    def retrieve(self, query: str, k: int = 10):
        _ = k
        assert query == "Explain Miranda warnings"
        return [
            LegalChunk(
                id="chunk_scotus",
                doc_id="doc_scotus",
                text="Miranda warnings are required during custodial interrogation.",
                chunk_index=0,
                doc_type="case",
                court_level="scotus",
                citation="384 U.S. 436",
            ),
            LegalChunk(
                id="chunk_district",
                doc_id="doc_district",
                text="A district court discussed exceptions to Miranda warnings.",
                chunk_index=1,
                doc_type="case",
                court_level="district",
            ),
        ]


class _FakeVerifier:
    def __init__(self):
        self.calls = []

    def verify_claims_batch(self, claims, chunks):
        self.calls.append((len(claims), [chunk.id for chunk in chunks]))
        return [
            AggregatedScore(
                final_score=0.93,
                is_contradicted=False,
                best_chunk_idx=0,
                best_chunk=chunks[0],
                support_ratio=1.0,
                component_scores={"contradiction_max": 0.02},
            )
            for _ in claims
        ]


class _FakeClaim:
    text = "Miranda warnings are required during custodial interrogation."

    def to_dict(self):
        return {"text": self.text}


def test_pipeline_uses_rag_generation_when_retrieval_is_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")
    monkeypatch.setattr(pipeline_module, "decompose_document", lambda doc: [_FakeClaim()])

    llm = _FakeLLM()
    verifier = _FakeVerifier()
    pipeline = QueryPipeline(
        db=db,
        llm=llm,
        retriever=_FakeRetriever(),
        verifier=verifier,
    )

    result = pipeline.run(user_id=user["id"], query="Explain Miranda warnings")

    meta = result["pipeline"]
    interaction = result["interaction"]
    assistant = result["assistant_message"]

    assert meta["retrieval_used"] is True
    assert meta["generation_mode"] == "rag"
    assert meta["retrieval_backend_status"] == "configured"
    assert meta["retrieval_chunk_count"] == 2
    assert meta["verification_backend_status"] == "ok"
    assert meta["claims"][0]["verification"]["verdict"] == "VERIFIED"
    assert meta["claims"][0]["verification"]["best_chunk"]["citation"] == "384 U.S. 436"
    assert interaction["query"] == "Explain Miranda warnings"
    assert assistant["interaction_id"] == interaction["id"]
    assert llm.direct_queries == []
    assert len(llm.context_calls) == 1
    assert "Citation: 384 U.S. 436" in llm.context_calls[0][1][0]
    assert verifier.calls == [(1, ["chunk_scotus", "chunk_district"])]

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
        assert claim_rows[0]["verdict"] == "VERIFIED"

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


def test_pipeline_uses_direct_generation_without_retrieval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = Database(tmp_path / "pipeline.db")
    db.initialize()
    user = db.create_user("alice", "hashed")
    monkeypatch.setattr(pipeline_module, "decompose_document", lambda doc: [_FakeClaim()])

    monkeypatch.setattr(
        pipeline_module,
        "_load_default_retriever",
        lambda: (None, "unavailable:no_indices"),
    )

    llm = _FakeLLM()
    pipeline = QueryPipeline(db=db, llm=llm, retriever=None, verifier=_FakeVerifier())

    result = pipeline.run(user_id=user["id"], query="Explain Miranda warnings")

    meta = result["pipeline"]
    assert meta["retrieval_used"] is False
    assert meta["generation_mode"] == "direct"
    assert meta["retrieval_chunk_count"] == 0
    assert meta["verification_backend_status"] == "skipped:no_retriever"
    assert llm.direct_queries == ["Explain Miranda warnings"]
    assert llm.context_calls == []


def test_load_default_retriever_wires_cross_encoder_reranker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def fake_load_bm25_index(index_path: Path):
        captured["bm25_path"] = index_path
        return object()

    def fake_load_vector_store(chroma_path: Path):
        captured["chroma_path"] = chroma_path
        return object(), object()

    class _FakeReranker:
        pass

    class _FakeHybridRetriever:
        def __init__(self, **kwargs):
            captured["retriever_kwargs"] = kwargs

    monkeypatch.setattr(pipeline_module, "_load_bm25_index", fake_load_bm25_index)
    monkeypatch.setattr(pipeline_module, "_load_vector_store", fake_load_vector_store)
    monkeypatch.setattr(pipeline_module, "CrossEncoderReranker", _FakeReranker)
    monkeypatch.setattr(pipeline_module, "HybridRetriever", _FakeHybridRetriever)
    monkeypatch.setattr(pipeline_module, "INDEX_DIR", tmp_path)

    retriever, status = pipeline_module._load_default_retriever()

    assert status == "ok"
    assert isinstance(retriever, _FakeHybridRetriever)
    assert captured["bm25_path"] == tmp_path / "bm25.pkl"
    assert "reranker" in captured["retriever_kwargs"]
    assert isinstance(captured["retriever_kwargs"]["reranker"], _FakeReranker)
