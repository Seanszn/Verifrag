"""Integration tests over 10 real CourtListener documents."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from src.indexing.bm25_index import BM25Index
from src.indexing.chroma_store import ChromaStore
from src.indexing.index_builder import build_indices
from src.ingestion.chunker import chunk_document
from src.ingestion.corpus_preparer import load_raw_documents, prepare_corpus
from src.retrieval.hybrid_retriever import HybridRetriever


pytestmark = pytest.mark.smoke

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_SOURCE = PROJECT_ROOT / "data" / "raw" / "scotus_cases.jsonl"


class DeterministicKeywordEmbedder:
    """Offline lexical hash embedder for repeatable dense-index tests."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts, batch_size=32, normalize=True):
        _ = batch_size
        vectors = []
        for text in texts:
            vector = np.zeros(self.dim, dtype=np.float32)
            for token in _tokenize(text):
                digest = hashlib.sha1(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dim
                vector[index] += 1.0
            if normalize:
                norm = float(np.linalg.norm(vector))
                if norm > 0.0:
                    vector /= norm
            vectors.append(vector)
        return np.vstack(vectors) if vectors else np.empty((0, self.dim), dtype=np.float32)


def test_step_01_load_first_10_real_documents_and_record_inputs(tmp_path: Path):
    subset_raw = _write_real_subset(tmp_path, count=10)
    documents = load_raw_documents(subset_raw)

    report = {
        "source_file": str(subset_raw),
        "documents_loaded": len(documents),
        "doc_ids": [doc.id for doc in documents],
        "case_names": [doc.case_name for doc in documents[:3]],
        "first_document": {
            "id": documents[0].id,
            "case_name": documents[0].case_name,
            "citation": documents[0].citation,
            "court_level": documents[0].court_level,
            "word_count": len(documents[0].full_text.split()),
            "preview": documents[0].full_text[:600],
        },
    }
    report_path = _write_report(tmp_path, "step_01_load_inputs.json", report)

    assert len(documents) == 10
    assert documents[0].id.startswith("cl_")
    assert documents[0].doc_type == "case"
    assert report_path.exists()


def test_step_02_chunk_first_real_document_and_record_outputs(tmp_path: Path):
    subset_raw = _write_real_subset(tmp_path, count=10)
    documents = load_raw_documents(subset_raw)
    first = documents[0]
    chunks = chunk_document(first, chunk_size=80, chunk_overlap=20)

    report = {
        "input": {
            "doc_id": first.id,
            "case_name": first.case_name,
            "citation": first.citation,
            "word_count": len(first.full_text.split()),
        },
        "output": {
            "chunk_count": len(chunks),
            "first_chunks": [
                {
                    "id": chunk.id,
                    "chunk_index": chunk.chunk_index,
                    "word_count": len(chunk.text.split()),
                    "preview": chunk.text[:400],
                }
                for chunk in chunks[:3]
            ],
            "last_chunk": {
                "id": chunks[-1].id,
                "chunk_index": chunks[-1].chunk_index,
                "word_count": len(chunks[-1].text.split()),
                "preview": chunks[-1].text[:250],
            },
        },
    }
    report_path = _write_report(tmp_path, "step_02_chunk_outputs.json", report)

    assert len(chunks) > 1
    assert chunks[0].id == f"{first.id}:0"
    assert all(chunk.doc_id == first.id for chunk in chunks)
    assert report_path.exists()


def test_step_03_prepare_real_10_doc_subset_and_record_processed_outputs(tmp_path: Path):
    subset_raw = _write_real_subset(tmp_path, count=10)
    raw_dir = subset_raw.parent
    processed_dir = tmp_path / "processed"
    summary = prepare_corpus(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        raw_files=[subset_raw],
        chunk_size=80,
        chunk_overlap=20,
    )

    processed_file = processed_dir / "scotus_chunks.jsonl"
    processed_rows = _read_jsonl(processed_file)
    summary_payload = json.loads((processed_dir / "prep_summary.json").read_text(encoding="utf-8"))

    report = {
        "input": {
            "raw_file": str(subset_raw),
            "documents_requested": 10,
        },
        "output": {
            "documents_loaded": summary.documents_loaded,
            "chunks_written": summary.chunks_written,
            "processed_file": str(processed_file),
            "summary_file": str(processed_dir / "prep_summary.json"),
            "first_processed_rows": processed_rows[:3],
        },
    }
    report_path = _write_report(tmp_path, "step_03_prepare_outputs.json", report)

    assert summary.documents_loaded == 10
    assert summary.chunks_written == len(processed_rows)
    assert len({row["doc_id"] for row in processed_rows}) == 10
    assert summary_payload["documents_loaded"] == 10
    assert report_path.exists()


def test_step_04_build_indices_and_retrieve_over_real_10_doc_subset(tmp_path: Path):
    subset_raw = _write_real_subset(tmp_path, count=10)
    raw_dir = subset_raw.parent
    documents = load_raw_documents(subset_raw)
    processed_dir = tmp_path / "processed"
    index_dir = tmp_path / "index"
    prepare_corpus(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        raw_files=[subset_raw],
        chunk_size=80,
        chunk_overlap=20,
    )

    embedder = DeterministicKeywordEmbedder(dim=256)
    artifacts = build_indices(
        processed_dir=processed_dir,
        processed_files=[processed_dir / "scotus_chunks.jsonl"],
        output_dir=index_dir,
        embedder=embedder,
        collection_name="real_10_doc_chunks",
    )

    bm25 = BM25Index(index_path=artifacts.bm25_path)
    bm25.load()
    vector_store = ChromaStore(path=artifacts.chroma_path, collection_name="real_10_doc_chunks")
    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25,
        embedder=embedder,
    )

    query = documents[0].case_name or documents[0].id
    dense_hits = vector_store.search(embedder.encode([query])[0], k=5)
    sparse_hits = bm25.search(query, k=5)
    fused_hits = retriever.retrieve(query, k=5)

    report = {
        "input": {
            "processed_file": str(processed_dir / "scotus_chunks.jsonl"),
            "query": query,
        },
        "output": {
            "chunk_count": artifacts.chunk_count,
            "embedding_shape": list(artifacts.embedding_shape),
            "bm25_top_5": [
                {"chunk_id": chunk_id, "score": score, "doc_id": metadata["doc_id"], "preview": metadata["text"][:180]}
                for chunk_id, score, metadata in sparse_hits
            ],
            "dense_top_5": [
                {"chunk_id": chunk_id, "score": score, "doc_id": metadata["doc_id"], "preview": metadata["text"][:180]}
                for chunk_id, score, metadata in dense_hits
            ],
            "fused_top_5": [
                {"chunk_id": chunk.id, "doc_id": chunk.doc_id, "preview": chunk.text[:180]}
                for chunk in fused_hits
            ],
        },
    }
    report_path = _write_report(tmp_path, "step_04_index_and_retrieve_outputs.json", report)

    assert artifacts.chunk_count > 0
    assert sparse_hits
    assert dense_hits
    assert fused_hits
    assert fused_hits[0].doc_id == documents[0].id
    assert report_path.exists()


def _write_real_subset(tmp_path: Path, *, count: int) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / "scotus_cases.jsonl"

    lines = [
        line
        for line in RAW_SOURCE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:count]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_report(tmp_path: Path, filename: str, payload: dict) -> Path:
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / filename
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
