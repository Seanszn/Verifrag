"""
Build Index - Create ChromaDB/BM25 indices from processed documents.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

# Add project root to import path when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import INDEX_DIR, PROCESSED_DIR, RAW_DIR, RETRIEVAL, VECTOR_STORE
from src.indexing.bm25_index import BM25Index
from src.indexing.chroma_store import ChromaStore
from src.indexing.embedder import Embedder
from src.ingestion.document import LegalChunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ChromaDB and BM25 indices from local JSONL data.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory containing chunk JSONL files. If empty, raw documents are used as a fallback.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Fallback directory containing raw document JSONL files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=VECTOR_STORE.chroma_batch_size,
        help="Batch size used for embedding generation and Chroma inserts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of chunks to index.",
    )
    return parser.parse_args()


def iter_jsonl_records(directory: Path):
    if not directory.exists():
        return

    for path in sorted(directory.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def load_chunks_from_dir(directory: Path, limit: int | None = None) -> list[LegalChunk]:
    chunks: list[LegalChunk] = []
    for record in iter_jsonl_records(directory) or []:
        chunks.extend(_chunks_from_record(record))
        if limit is not None and len(chunks) >= limit:
            return chunks[:limit]
    return chunks


def _chunks_from_record(record: dict) -> list[LegalChunk]:
    if {"id", "doc_id", "text", "chunk_index", "doc_type"}.issubset(record):
        return [_chunk_from_dict(record)]

    if {"id", "doc_type", "full_text"}.issubset(record):
        return _chunk_document(record)

    return []


def _chunk_from_dict(record: dict) -> LegalChunk:
    raw_date = record.get("date_decided")
    parsed_date = date.fromisoformat(raw_date) if raw_date else None
    return LegalChunk(
        id=str(record["id"]),
        doc_id=str(record["doc_id"]),
        text=str(record["text"]),
        chunk_index=int(record["chunk_index"]),
        doc_type=str(record["doc_type"]),
        court_level=record.get("court_level"),
        citation=record.get("citation"),
        date_decided=parsed_date,
        section_type=record.get("section_type"),
    )


def _chunk_document(record: dict) -> list[LegalChunk]:
    text = " ".join(str(record.get("full_text", "")).split())
    if not text:
        return []

    words = text.split()
    chunk_size = max(32, int(RETRIEVAL.chunk_size))
    overlap = max(0, min(int(RETRIEVAL.chunk_overlap), chunk_size - 1))
    step = max(1, chunk_size - overlap)
    chunks: list[LegalChunk] = []

    for idx, start in enumerate(range(0, len(words), step)):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            continue
        chunk_text = " ".join(chunk_words)
        chunks.append(
            LegalChunk(
                id=f"{record['id']}:{idx}",
                doc_id=str(record["id"]),
                text=chunk_text,
                chunk_index=idx,
                doc_type=str(record["doc_type"]),
                court_level=record.get("court_level"),
                citation=record.get("citation"),
                date_decided=date.fromisoformat(record["date_decided"]) if record.get("date_decided") else None,
                section_type=record.get("section_type"),
            )
        )

    return chunks


def build_indices(chunks: list[LegalChunk], batch_size: int) -> tuple[int, tuple[int, int]]:
    texts = [chunk.text for chunk in chunks]
    embedder = Embedder()
    embeddings = embedder.encode(texts, batch_size=batch_size, normalize=True)

    vector_store = ChromaStore(batch_size=batch_size)
    chunk_ids = [chunk.id for chunk in chunks]
    vector_store.delete(chunk_ids)
    vector_store.add(chunk_ids, embeddings, [chunk.to_dict() for chunk in chunks])
    vector_store.save()

    bm25 = BM25Index(index_path=INDEX_DIR / "bm25.pkl")
    bm25.build(chunks)
    bm25.save()

    return len(chunks), embeddings.shape


def main() -> None:
    args = parse_args()
    chunks = load_chunks_from_dir(args.input_dir, limit=args.limit)
    source_dir = args.input_dir

    if not chunks:
        chunks = load_chunks_from_dir(args.raw_dir, limit=args.limit)
        source_dir = args.raw_dir

    if not chunks:
        raise SystemExit(
            f"No chunkable JSONL records found in {args.input_dir} or {args.raw_dir}"
        )

    total_chunks, vector_shape = build_indices(chunks, batch_size=max(1, args.batch_size))
    print(f"Loaded chunks from: {source_dir}")
    print(f"Indexed chunks: {total_chunks}")
    print(f"Embedding matrix shape: {vector_shape}")
    print(f"Chroma path: {VECTOR_STORE.chroma_path}")
    print(f"BM25 path: {INDEX_DIR / 'bm25.pkl'}")


if __name__ == "__main__":
    main()
