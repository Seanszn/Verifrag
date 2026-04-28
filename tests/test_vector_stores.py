"""Tests for ChromaDB vector store backend."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.indexing import chroma_store as chroma_store_module
from src.indexing.chroma_store import ChromaStore
from src.ingestion.document import LegalChunk


pytestmark = pytest.mark.smoke


def _chunk(chunk_id: str, text: str, idx: int) -> LegalChunk:
    return LegalChunk(
        id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        text=text,
        chunk_index=idx,
        doc_type="case",
        court_level="scotus",
        citation="123 U.S. 456",
    )


class _FakeCollection:
    def __init__(self):
        self.rows: dict[str, dict] = {}

    def add(self, *, ids, embeddings, documents, metadatas):
        for chunk_id, embedding, document, metadata in zip(ids, embeddings, documents, metadatas):
            self.rows[str(chunk_id)] = {
                "embedding": np.asarray(embedding, dtype=np.float32),
                "document": document,
                "metadata": dict(metadata),
            }

    def query(self, *, query_embeddings, n_results, include):
        _ = include
        query = np.asarray(query_embeddings[0], dtype=np.float32)
        ranked = []
        for chunk_id, row in self.rows.items():
            distance = float(1.0 - np.dot(query, row["embedding"]))
            ranked.append((distance, chunk_id, row))
        ranked.sort(key=lambda item: item[0])
        ranked = ranked[:n_results]

        return {
            "ids": [[chunk_id for _, chunk_id, _ in ranked]],
            "documents": [[row["document"] for _, _, row in ranked]],
            "metadatas": [[dict(row["metadata"]) for _, _, row in ranked]],
            "distances": [[distance for distance, _, _ in ranked]],
        }

    def delete(self, *, ids):
        for chunk_id in ids:
            self.rows.pop(str(chunk_id), None)

    def count(self):
        return len(self.rows)


class _FakePersistentClient:
    _collections: dict[tuple[str, str], _FakeCollection] = {}

    def __init__(self, *, path: str):
        self.path = path

    def get_or_create_collection(self, *, name: str, metadata: dict):
        _ = metadata
        key = (self.path, name)
        if key not in self._collections:
            self._collections[key] = _FakeCollection()
        return self._collections[key]


def _install_fake_chromadb(monkeypatch):
    _FakePersistentClient._collections = {}
    monkeypatch.setattr(
        chroma_store_module,
        "chromadb",
        SimpleNamespace(PersistentClient=_FakePersistentClient),
    )


def test_chroma_store_search_returns_ranked_results_with_round_tripped_metadata(tmp_path, monkeypatch):
    _install_fake_chromadb(monkeypatch)
    store = ChromaStore(path=tmp_path / "chroma", collection_name="search_test", batch_size=2)
    chunk_a = _chunk("a", "warrant required for phone search", 0)
    chunk_b = _chunk("b", "tax venue transfer dispute", 1)
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    store.add(
        [chunk_a.id, chunk_b.id],
        embeddings,
        [chunk_a.to_dict(), chunk_b.to_dict()],
    )

    results = store.search(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), k=2)

    assert [chunk_id for chunk_id, _, _ in results] == ["a", "b"]
    assert results[0][1] > results[1][1]
    assert results[0][2]["text"] == chunk_a.text
    assert results[0][2]["doc_id"] == chunk_a.doc_id
    assert results[0][2]["chunk_index"] == chunk_a.chunk_index


def test_chroma_store_delete_removes_vectors(tmp_path, monkeypatch):
    _install_fake_chromadb(monkeypatch)
    store = ChromaStore(path=tmp_path / "chroma", collection_name="delete_test")
    chunk_a = _chunk("a", "alpha", 0)
    chunk_b = _chunk("b", "beta", 1)
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    store.add(
        [chunk_a.id, chunk_b.id],
        embeddings,
        [chunk_a.to_dict(), chunk_b.to_dict()],
    )

    store.delete([chunk_a.id])
    results = store.search(np.asarray([1.0, 0.0], dtype=np.float32), k=5)

    assert [chunk_id for chunk_id, _, _ in results] == ["b"]


def test_chroma_store_persists_across_instances(tmp_path, monkeypatch):
    _install_fake_chromadb(monkeypatch)
    path = tmp_path / "chroma"
    first = ChromaStore(path=path, collection_name="persist_test")
    chunk = _chunk("persisted", "precedent remains binding", 0)
    first.add(
        [chunk.id],
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        [chunk.to_dict()],
    )
    first.save()

    second = ChromaStore(path=path, collection_name="persist_test")
    results = second.search(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), k=1)

    assert len(results) == 1
    assert results[0][0] == chunk.id
    assert results[0][2]["text"] == chunk.text
