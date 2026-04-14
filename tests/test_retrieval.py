"""Tests for hybrid retrieval and reranking."""

from __future__ import annotations

import numpy as np
import pytest

from src.indexing.bm25_index import BM25Index
from src.ingestion.document import LegalChunk
from src.retrieval.case_targeting import (
    case_names_match,
    dedupe_search_hits_by_canonical_identity,
    extract_target_case_name,
    filter_search_hits_to_target_case,
)
from src.retrieval.hybrid_retriever import HybridRetriever


pytestmark = pytest.mark.smoke


class _FakeEmbedder:
    def __init__(self, vector: np.ndarray):
        self.vector = vector
        self.seen_texts = []

    def encode(self, texts, batch_size=32, normalize=True):
        _ = batch_size, normalize
        self.seen_texts.extend(texts)
        return np.asarray([self.vector for _ in texts], dtype=np.float32)


class _FakeVectorStore:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, query_embedding, k):
        self.calls.append((query_embedding, k))
        return self.hits[:k]


class _FakeSparseIndex:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, query, k):
        self.calls.append((query, k))
        return self.hits[:k]


class _FakeReranker:
    def __init__(self, score_map):
        self.score_map = score_map
        self.calls = []

    def score(self, query, chunks):
        self.calls.append((query, [chunk.id for chunk in chunks]))
        return [self.score_map[chunk.id] for chunk in chunks]


def _chunk(chunk_id: str, text: str, idx: int) -> LegalChunk:
    return LegalChunk(
        id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        text=text,
        chunk_index=idx,
        doc_type="case",
        court_level="scotus",
    )


def _hit(chunk: LegalChunk, score: float):
    return (chunk.id, score, chunk.to_dict())


def _metadata_hit(
    chunk_id: str,
    doc_id: str,
    text: str,
    idx: int,
    score: float,
    *,
    case_name: str | None = None,
    court_level: str = "scotus",
    date_decided: str = "2026-02-20",
    citation: str | None = None,
):
    return (
        chunk_id,
        score,
        {
            "id": chunk_id,
            "doc_id": doc_id,
            "text": text,
            "chunk_index": idx,
            "doc_type": "case",
            "case_name": case_name,
            "court_level": court_level,
            "date_decided": date_decided,
            "citation": citation,
        },
    )


def test_bm25_index_returns_matching_chunks_by_score(tmp_path):
    chunk_a = _chunk("a", "warrant required for cellphone search", 0)
    chunk_b = _chunk("b", "search search search warrant exception", 1)
    chunk_c = _chunk("c", "tax dispute and venue transfer", 2)

    index = BM25Index(save_path=str(tmp_path / "bm25.pkl"))
    index.build(
        [chunk_a.text, chunk_b.text, chunk_c.text],
        [chunk_a.to_dict(), chunk_b.to_dict(), chunk_c.to_dict()],
    )

    results = index.search("search warrant", k=2)

    assert [chunk_id for chunk_id, _, _ in results] == ["1", "0"]
    assert all(score > 0 for _, score, _ in results)


def test_hybrid_retriever_fuses_dense_and_sparse_hits_with_rrf():
    chunk_a = _chunk("a", "alpha", 0)
    chunk_b = _chunk("b", "beta", 1)
    chunk_c = _chunk("c", "gamma", 2)

    vector_store = _FakeVectorStore([_hit(chunk_a, 0.95), _hit(chunk_c, 0.90)])
    sparse_index = _FakeSparseIndex([_hit(chunk_b, 12.0), _hit(chunk_a, 8.0)])
    embedder = _FakeEmbedder(np.array([1.0, 0.0, 0.0], dtype=np.float32))

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=sparse_index,
        embedder=embedder,
        rrf_k=60,
        reranker=None,
    )

    results = retriever.retrieve("warrant query", k=3)

    assert [chunk.id for chunk in results] == ["a", "b", "c"]
    assert embedder.seen_texts == ["warrant query"]
    assert vector_store.calls[0][0].shape == (3,)
    assert sparse_index.calls == [("warrant query", 20)]


def test_hybrid_retriever_supports_legacy_top_k_dictionary_api():
    vector_store = _FakeVectorStore(
        [
            (
                "doc_2",
                0.95,
                {
                    "text": "The Supreme Court ruled on the family trust taxation.",
                    "source_file": "trust.pdf",
                    "doc_type": "case",
                    "chunk_index": 0,
                },
            )
        ]
    )
    sparse_index = _FakeSparseIndex(
        [
            (
                "doc_2",
                12.0,
                {
                    "text": "The Supreme Court ruled on the family trust taxation.",
                    "source_file": "trust.pdf",
                    "doc_type": "case",
                    "chunk_index": 0,
                },
            )
        ]
    )
    embedder = _FakeEmbedder(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    retriever = HybridRetriever(embedder, vector_store, sparse_index)

    results = retriever.retrieve(query="Supreme Court family trust", top_k=2)

    assert results[0]["id"] == "doc_2"
    assert "family trust" in results[0]["text"]
    assert results[0]["source"] == "trust.pdf"
    assert results[0]["rrf_score"] > 0.0


def test_hybrid_retriever_reranker_can_override_fused_order():
    chunk_a = _chunk("a", "alpha", 0)
    chunk_b = _chunk("b", "beta", 1)
    chunk_c = _chunk("c", "gamma", 2)

    vector_store = _FakeVectorStore([_hit(chunk_a, 0.95), _hit(chunk_b, 0.92), _hit(chunk_c, 0.90)])
    sparse_index = _FakeSparseIndex([])
    embedder = _FakeEmbedder(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    reranker = _FakeReranker({"a": 0.1, "b": 0.2, "c": 0.9})

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=sparse_index,
        embedder=embedder,
        reranker=reranker,
        rerank_k=3,
    )

    results = retriever.retrieve("query", k=2)

    assert [chunk.id for chunk in results] == ["c", "b"]
    assert reranker.calls == [("query", ["a", "b", "c"])]


def test_hybrid_retriever_returns_empty_for_blank_query():
    vector_store = _FakeVectorStore([])
    sparse_index = _FakeSparseIndex([])
    embedder = _FakeEmbedder(np.array([1.0], dtype=np.float32))

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=sparse_index,
        embedder=embedder,
    )

    assert retriever.retrieve("   ", k=5) == []
    assert vector_store.calls == []
    assert sparse_index.calls == []


def test_case_targeting_handles_revision_suffixes_and_prefers_exact_variant():
    query = (
        "In Learning Resources, Inc. v. Trump Revisions: 2/23/26, "
        "what did the Supreme Court decide?"
    )
    extracted = extract_target_case_name(query)

    assert extracted == "Learning Resources, Inc. v. Trump Revisions: 2/23/26"
    assert case_names_match(extracted, "Learning Resources, Inc. v. Trump")
    assert not case_names_match("Alpha v. Beta", "Alpha v. Beta Holdings")

    hits = [
        _metadata_hit(
            "base:0",
            "doc_base",
            "Fulfilling that role, we hold that IEEPA does not authorize the President to impose tariffs.",
            0,
            0.95,
            case_name="Learning Resources, Inc. v. Trump",
        ),
        _metadata_hit(
            "rev:0",
            "doc_rev",
            "Fulfilling that role, we hold that IEEPA does not authorize the President to impose tariffs.",
            0,
            0.90,
            case_name="Learning Resources, Inc. v. Trump Revisions: 2/23/26",
        ),
    ]

    filtered = filter_search_hits_to_target_case(
        query,
        hits,
        metadata_case_name=lambda item: item[2]["case_name"],
    )
    deduped = dedupe_search_hits_by_canonical_identity(
        filtered,
        metadata_doc_id=lambda item: item[2]["doc_id"],
        metadata_chunk_index=lambda item: item[2]["chunk_index"],
        metadata_text=lambda item: item[2]["text"],
        metadata_case_name=lambda item: item[2]["case_name"],
        metadata_citation=lambda item: item[2]["citation"],
        metadata_date_decided=lambda item: item[2]["date_decided"],
        metadata_court_level=lambda item: item[2]["court_level"],
    )

    assert [item[2]["doc_id"] for item in filtered] == ["doc_rev", "doc_base"]
    assert [item[2]["doc_id"] for item in deduped] == ["doc_rev"]


def test_filter_search_hits_returns_empty_for_off_target_case_query():
    query = "In Alpha v. Beta, what did the Supreme Court decide?"
    hits = [
        _metadata_hit(
            "other:0",
            "doc_other",
            "An unrelated case discussed venue transfer.",
            0,
            1.0,
            case_name="Gamma v. Delta",
        ),
    ]

    filtered = filter_search_hits_to_target_case(
        query,
        hits,
        metadata_case_name=lambda item: item[2]["case_name"],
    )

    assert filtered == []


def test_hybrid_retriever_scopes_and_dedupes_revision_case_variants():
    query = (
        "In Learning Resources, Inc. v. Trump Revisions: 2/23/26, "
        "what did the Supreme Court decide?"
    )
    exact_case = "Learning Resources, Inc. v. Trump Revisions: 2/23/26"
    base_case = "Learning Resources, Inc. v. Trump"
    text = "Fulfilling that role, we hold that IEEPA does not authorize the President to impose tariffs."

    vector_store = _FakeVectorStore(
        [
            _metadata_hit("base:0", "doc_base", text, 0, 0.95, case_name=base_case),
            _metadata_hit(
                "other:0",
                "doc_other",
                "Miranda warnings are required during custodial interrogation.",
                0,
                0.93,
                case_name="Miranda v. Arizona",
            ),
        ]
    )
    sparse_index = _FakeSparseIndex(
        [
            _metadata_hit("rev:0", "doc_rev", text, 0, 12.0, case_name=exact_case),
            _metadata_hit(
                "rev:1",
                "doc_rev",
                "The Court today nonetheless concludes otherwise.",
                1,
                11.0,
                case_name=exact_case,
            ),
        ]
    )
    embedder = _FakeEmbedder(np.array([1.0, 0.0], dtype=np.float32))

    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=sparse_index,
        embedder=embedder,
        reranker=None,
    )

    results = retriever.retrieve(query, k=3)

    assert [chunk.id for chunk in results] == ["rev:0", "rev:1"]
