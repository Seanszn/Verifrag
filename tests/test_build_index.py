"""Tests for building persisted retrieval indices from processed chunks."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.build_index import main as build_index_main
from src.indexing import chroma_store as chroma_store_module
from src.indexing.bm25_index import BM25Index
from src.indexing.chroma_store import ChromaStore
from src.indexing import index_builder as index_builder_module
from src.indexing.index_builder import build_indices


pytestmark = pytest.mark.smoke


class _FakeEmbedder:
    def encode(self, texts, batch_size=32, normalize=True):
        _ = batch_size, normalize
        vectors = []
        for text in texts:
            if "warrant" in text.lower():
                vectors.append([1.0, 0.0, 0.0])
            else:
                vectors.append([0.0, 1.0, 0.0])
        return np.asarray(vectors, dtype=np.float32)


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


def _write_processed_file(path: Path) -> None:
    rows = [
        {
            "id": "cl_1:0",
            "doc_id": "cl_1",
            "text": "The warrant requirement governs cellphone searches.",
            "chunk_index": 0,
            "doc_type": "case",
            "court_level": "scotus",
            "citation": "573 U.S. 373",
            "date_decided": "2014-06-25",
        },
        {
            "id": "cl_2:0",
            "doc_id": "cl_2",
            "text": "Venue transfer disputes are analyzed under Section 1404.",
            "chunk_index": 0,
            "doc_type": "case",
            "court_level": "circuit",
            "citation": "487 U.S. 22",
            "date_decided": "1988-05-31",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_build_indices_creates_persisted_bm25_chroma_and_summary(tmp_path, monkeypatch):
    _install_fake_chromadb(monkeypatch)
    processed_dir = tmp_path / "processed"
    processed_file = processed_dir / "sample_chunks.jsonl"
    _write_processed_file(processed_file)

    artifacts = build_indices(
        processed_dir=processed_dir,
        output_dir=tmp_path / "index",
        collection_name="test_chunks",
        embedder=_FakeEmbedder(),
    )

    bm25 = BM25Index(index_path=artifacts.bm25_path)
    bm25.load()
    bm25_results = bm25.search("warrant search", k=2)
    assert bm25_results[0][0] == "cl_1:0"

    store = ChromaStore(path=artifacts.chroma_path, collection_name="test_chunks")
    dense_results = store.search(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), k=2)
    assert dense_results[0][0] == "cl_1:0"
    assert dense_results[0][2]["doc_id"] == "cl_1"

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["chunk_count"] == 2
    assert summary["embedding_shape"] == [2, 3]
    assert summary["collection_name"] == "test_chunks"
    assert summary["processed_files"] == [str(processed_file.resolve())]


def test_build_index_cli_writes_default_named_artifacts(tmp_path, monkeypatch, capsys):
    _install_fake_chromadb(monkeypatch)
    monkeypatch.setattr(index_builder_module, "Embedder", _FakeEmbedder)
    processed_dir = tmp_path / "processed"
    _write_processed_file(processed_dir / "alpha_chunks.jsonl")

    exit_code = build_index_main(
        [
            "--processed-dir",
            str(processed_dir),
            "--output-dir",
            str(tmp_path / "index"),
            "--collection-name",
            "cli_chunks",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Indexed 2 chunks from 1 file(s)." in captured.out
    assert (tmp_path / "index" / "bm25.pkl").exists()
    assert (tmp_path / "index" / "index_summary.json").exists()
