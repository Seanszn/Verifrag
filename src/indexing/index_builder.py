"""Helpers for building dense and sparse indices from processed chunk JSONL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.config import INDEX_DIR, MODELS, PROCESSED_DIR, VECTOR_STORE
from src.indexing.bm25_index import BM25Index
from src.indexing.chroma_store import ChromaStore
from src.indexing.embedder import Embedder
from src.ingestion.document import LegalChunk


@dataclass(frozen=True)
class BuildArtifacts:
    processed_dir: Path
    processed_files: list[Path]
    chunk_count: int
    embedding_shape: tuple[int, int]
    chroma_path: Path
    bm25_path: Path
    collection_name: str
    summary_path: Path


def load_chunks(processed_files: Sequence[str | Path] | Iterable[str | Path]) -> list[LegalChunk]:
    """Load chunk rows from one or more processed JSONL files."""
    chunks: list[LegalChunk] = []
    for raw_path in processed_files:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                payload = line.strip()
                if not payload:
                    continue
                try:
                    row = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
                chunks.append(_chunk_from_row(row))
    return chunks


def find_processed_files(processed_dir: str | Path, pattern: str = "*.jsonl") -> list[Path]:
    """Return sorted processed chunk files under a directory."""
    root = Path(processed_dir)
    return sorted(path for path in root.glob(pattern) if path.is_file())


def build_indices(
    *,
    processed_dir: str | Path = PROCESSED_DIR,
    processed_files: Sequence[str | Path] | None = None,
    output_dir: str | Path = INDEX_DIR,
    bm25_filename: str = "bm25.pkl",
    chroma_dirname: str = "chroma",
    summary_filename: str = "index_summary.json",
    collection_name: str | None = None,
    embedder: Embedder | None = None,
) -> BuildArtifacts:
    """Build BM25 and Chroma indices from processed chunk JSONL files."""
    processed_root = Path(processed_dir)
    files = [Path(item) for item in (processed_files or find_processed_files(processed_root))]
    if not files:
        raise ValueError(f"No processed chunk files found under {processed_root}")

    chunks = load_chunks(files)
    if not chunks:
        raise ValueError("Processed chunk files did not contain any chunks")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    bm25_path = output_root / bm25_filename
    chroma_path = output_root / chroma_dirname
    summary_path = output_root / summary_filename
    resolved_collection = collection_name or VECTOR_STORE.chroma_collection

    texts = [chunk.text for chunk in chunks]
    encoder = embedder or Embedder()
    embeddings = encoder.encode(texts, batch_size=VECTOR_STORE.chroma_batch_size, normalize=True)

    bm25 = BM25Index(chunks, index_path=bm25_path)
    bm25.save()

    store = ChromaStore(
        path=chroma_path,
        collection_name=resolved_collection,
        batch_size=VECTOR_STORE.chroma_batch_size,
        distance=VECTOR_STORE.chroma_distance,
    )
    store.delete([chunk.id for chunk in chunks])
    store.add(
        [chunk.id for chunk in chunks],
        embeddings,
        [chunk.to_dict() for chunk in chunks],
    )
    store.save()

    artifacts = BuildArtifacts(
        processed_dir=processed_root.resolve(),
        processed_files=[path.resolve() for path in files],
        chunk_count=len(chunks),
        embedding_shape=tuple(int(item) for item in embeddings.shape),
        chroma_path=chroma_path.resolve(),
        bm25_path=bm25_path.resolve(),
        collection_name=resolved_collection,
        summary_path=summary_path.resolve(),
    )
    _write_summary(artifacts)
    return artifacts


def _write_summary(artifacts: BuildArtifacts) -> None:
    payload = {
        "processed_dir": str(artifacts.processed_dir),
        "processed_files": [str(path) for path in artifacts.processed_files],
        "chunk_count": artifacts.chunk_count,
        "embedding_shape": list(artifacts.embedding_shape),
        "embedding_model": MODELS.embedding_model,
        "chroma_path": str(artifacts.chroma_path),
        "bm25_path": str(artifacts.bm25_path),
        "collection_name": artifacts.collection_name,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    artifacts.summary_path.parent.mkdir(parents=True, exist_ok=True)
    with artifacts.summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _chunk_from_row(row: dict) -> LegalChunk:
    raw_date = row.get("date_decided")
    parsed_date = None
    if raw_date:
        try:
            parsed_date = date.fromisoformat(str(raw_date))
        except ValueError:
            parsed_date = None

    return LegalChunk(
        id=row["id"],
        doc_id=row["doc_id"],
        text=row["text"],
        chunk_index=int(row["chunk_index"]),
        doc_type=row["doc_type"],
        case_name=row.get("case_name"),
        court=row.get("court"),
        court_level=row.get("court_level"),
        citation=row.get("citation"),
        date_decided=parsed_date,
        title=row.get("title"),
        section=row.get("section"),
        source_file=row.get("source_file"),
    )
