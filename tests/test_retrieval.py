"""Tests for hybrid retrieval."""

from __future__ import annotations

import numpy as np
import pytest

from src.indexing.bm25_index import BM25Index
from src.ingestion.document import LegalChunk
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_expansion import QueryExpansionConfig


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


class _QueryAwareSparseIndex:
    def __init__(self, default_hits, query_hits):
        self.default_hits = default_hits
        self.query_hits = query_hits
        self.calls = []

    def search(self, query, k):
        self.calls.append((query, k))
        for needle, hits in self.query_hits.items():
            if needle in query:
                return hits[:k]
        return self.default_hits[:k]


class _FakeReranker:
    def __init__(self, score_map):
        self.score_map = score_map
        self.calls = []

    def score(self, query, chunks):
        self.calls.append((query, [chunk.id for chunk in chunks]))
        return [self.score_map[chunk.id] for chunk in chunks]


class _ExplodingEmbedder:
    def encode(self, texts, batch_size=32, normalize=True):
        _ = texts, batch_size, normalize
        raise RuntimeError("dense model unavailable")


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


def test_bm25_index_returns_matching_chunks_by_score():
    chunk_a = _chunk("a", "warrant required for cellphone search", 0)
    chunk_b = _chunk("b", "search search search warrant exception", 1)
    chunk_c = _chunk("c", "tax dispute and venue transfer", 2)

    index = BM25Index([chunk_a, chunk_b, chunk_c])

    results = index.search("search warrant", k=2)

    assert [chunk_id for chunk_id, _, _ in results] == ["b", "a"]
    assert all(score > 0 for _, score, _ in results)


def test_bm25_index_supports_legacy_text_metadata_api():
    index = BM25Index()
    index.build(
        [
            "The quick brown fox jumps.",
            "This is a legal document regarding a family trust.",
        ],
        [{"id": 0}, {"id": 1}],
    )

    results = index.search("trust", k=1)

    assert len(results) == 1
    assert results[0][0] == "1"


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
    assert vector_store.calls[0][1] == 2
    assert sparse_index.calls == [("Supreme Court family trust", 2)]


def test_hybrid_retriever_backfills_missing_chunk_metadata_from_later_hits():
    dense_metadata = {
        "id": "burnett:0",
        "doc_id": "cl_10805632",
        "text": "Burnett dense chunk",
        "chunk_index": 0,
        "doc_type": "case",
        "court_level": "scotus",
    }
    sparse_metadata = {
        **dense_metadata,
        "case_name": "Burnett v. United States",
        "court": "scotus",
        "citation": "607 U. S. ____ (2026)",
    }

    vector_store = _FakeVectorStore([("burnett:0", 0.95, dense_metadata)])
    sparse_index = _FakeSparseIndex([("burnett:0", 12.0, sparse_metadata)])
    embedder = _FakeEmbedder(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    retriever = HybridRetriever(embedder=embedder, vector_store=vector_store, bm25_index=sparse_index)

    results = retriever.retrieve("Burnett query", k=1)

    assert len(results) == 1
    assert results[0].id == "burnett:0"
    assert results[0].case_name == "Burnett v. United States"
    assert results[0].court == "scotus"
    assert results[0].citation == "607 U. S. ____ (2026)"


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


def test_hybrid_retriever_falls_back_to_sparse_when_dense_search_fails():
    chunk_b = _chunk("b", "beta", 1)

    vector_store = _FakeVectorStore([_hit(chunk_b, 0.95)])
    sparse_index = _FakeSparseIndex([_hit(chunk_b, 12.0)])
    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=sparse_index,
        embedder=_ExplodingEmbedder(),
    )

    results = retriever.retrieve("warrant query", k=3)

    assert [chunk.id for chunk in results] == ["b"]
    assert vector_store.calls == []
    assert sparse_index.calls == [("warrant query", 20)]
    assert retriever.last_backend_status() == "warning:dense:RuntimeError"


def test_hybrid_retriever_expands_broad_research_query_and_reranks_with_original_query():
    chunk_a = _chunk("a", "standing requires a concrete dispute", 0)
    chunk_b = _chunk("b", "article iii injury in fact and redressability", 1)

    sparse_index = _QueryAwareSparseIndex(
        default_hits=[_hit(chunk_a, 12.0)],
        query_hits={"injury in fact": [_hit(chunk_b, 10.0)]},
    )
    reranker = _FakeReranker({"a": 0.1, "b": 0.9})
    retriever = HybridRetriever(
        bm25_index=sparse_index,
        reranker=reranker,
        query_expansion_config=QueryExpansionConfig(mode="rule", max_variants=4, max_terms=12),
    )
    query = "find cases about standing for environmental plaintiffs"

    results = retriever.retrieve(query, k=2)

    assert [chunk.id for chunk in results] == ["b", "a"]
    assert len(sparse_index.calls) > 1
    assert sparse_index.calls[0] == (query, 20)
    assert any("injury in fact" in call_query for call_query, _ in sparse_index.calls)
    assert reranker.calls == [(query, ["a", "b"])]
    meta = retriever.last_query_expansion_meta()
    assert meta["status"] == "applied"
    assert meta["sources"] == ["rule"]
    assert meta["original_query"] == query
    assert any("b" in item["sparse_chunk_ids"] for item in meta["retrieved_chunk_ids_per_variant"])


def test_hybrid_retriever_does_not_expand_named_authority_query():
    chunk_a = _chunk("a", "Miranda warning discussion", 0)
    sparse_index = _FakeSparseIndex([_hit(chunk_a, 12.0)])
    retriever = HybridRetriever(
        bm25_index=sparse_index,
        query_expansion_config=QueryExpansionConfig(mode="hybrid"),
    )
    query = "What did Miranda v. Arizona say about suppression?"

    results = retriever.retrieve(query, k=1)

    assert [chunk.id for chunk in results] == ["a"]
    assert sparse_index.calls == [(query, 20)]
    assert retriever.last_query_expansion_meta()["status"] == "not_applied:named_authority_query"


def test_hybrid_retriever_uses_safe_corpus_facets_for_corpus_aware_expansion():
    chunk_a = _chunk("a", "penalty discussion", 0)
    sparse_index = _FakeSparseIndex([_hit(chunk_a, 12.0)])
    retriever = HybridRetriever(
        bm25_index=sparse_index,
        query_expansion_config=QueryExpansionConfig(mode="corpus-aware", max_variants=3),
    )

    retriever.retrieve("find authorities about penalties", k=1)

    meta = retriever.last_query_expansion_meta()
    assert meta["status"] == "applied"
    assert meta["sources"] == ["corpus-aware"]
    assert "scotus" in meta["terms"]
    assert "case" in meta["terms"]
    assert all(" v. " not in variant for variant in meta["variants"])
