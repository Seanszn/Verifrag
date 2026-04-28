"""Prepare raw legal document JSONL files into processed chunk JSONL files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

from src.config import PROCESSED_DIR, RAW_DIR, RETRIEVAL
from src.ingestion.chunker import chunk_document
from src.ingestion.document import LegalDocument


@dataclass(frozen=True)
class PreparedFile:
    raw_file: Path
    output_file: Path
    documents_loaded: int
    chunks_written: int


@dataclass(frozen=True)
class PreparationSummary:
    raw_dir: Path
    processed_dir: Path
    raw_files: list[Path]
    prepared_files: list[PreparedFile]
    documents_loaded: int
    chunks_written: int
    summary_path: Path


def find_raw_files(raw_dir: str | Path, pattern: str = "*.jsonl") -> list[Path]:
    """Return sorted raw JSONL corpus files under a directory."""
    root = Path(raw_dir)
    return sorted(path for path in root.glob(pattern) if path.is_file())


def prepare_corpus(
    *,
    raw_dir: str | Path = RAW_DIR,
    processed_dir: str | Path = PROCESSED_DIR,
    raw_files: Sequence[str | Path] | None = None,
    summary_filename: str = "prep_summary.json",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> PreparationSummary:
    """Convert raw document JSONL files into processed chunk JSONL files."""
    resolved_raw_dir = Path(raw_dir)
    resolved_processed_dir = Path(processed_dir)
    resolved_processed_dir.mkdir(parents=True, exist_ok=True)

    files = [Path(item) for item in (raw_files or find_raw_files(resolved_raw_dir))]
    if not files:
        raise ValueError(f"No raw JSONL files found under {resolved_raw_dir}")

    size = int(chunk_size or RETRIEVAL.chunk_size)
    overlap = int(chunk_overlap or RETRIEVAL.chunk_overlap)
    prepared_files: list[PreparedFile] = []
    total_documents = 0
    total_chunks = 0

    for raw_file in files:
        documents = load_raw_documents(raw_file)
        output_file = resolved_processed_dir / _output_name_for(raw_file.name)
        chunks_written = write_processed_chunks(
            documents,
            output_file=output_file,
            chunk_size=size,
            chunk_overlap=overlap,
        )
        prepared_files.append(
            PreparedFile(
                raw_file=raw_file.resolve(),
                output_file=output_file.resolve(),
                documents_loaded=len(documents),
                chunks_written=chunks_written,
            )
        )
        total_documents += len(documents)
        total_chunks += chunks_written

    summary_path = resolved_processed_dir / summary_filename
    summary = PreparationSummary(
        raw_dir=resolved_raw_dir.resolve(),
        processed_dir=resolved_processed_dir.resolve(),
        raw_files=[path.resolve() for path in files],
        prepared_files=prepared_files,
        documents_loaded=total_documents,
        chunks_written=total_chunks,
        summary_path=summary_path.resolve(),
    )
    write_preparation_summary(summary)
    return summary


def load_raw_documents(raw_file: str | Path) -> list[LegalDocument]:
    """Load raw corpus-builder style JSONL records into LegalDocument objects."""
    path = Path(raw_file)
    documents: list[LegalDocument] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                row = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
            documents.append(_document_from_row(row))
    return documents


def write_processed_chunks(
    documents: Sequence[LegalDocument] | Iterable[LegalDocument],
    *,
    output_file: str | Path,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    """Chunk documents and write the resulting rows to one processed JSONL file."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    chunks_written = 0
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            chunks = chunk_document(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            for chunk in chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
                chunks_written += 1

    return chunks_written


def write_preparation_summary(summary: PreparationSummary) -> None:
    """Persist a compact summary matching the existing prep_summary shape."""
    payload = {
        "raw_dir": str(summary.raw_dir),
        "processed_dir": str(summary.processed_dir),
        "raw_files": [path.name for path in summary.raw_files],
        "documents_loaded": summary.documents_loaded,
        "chunks_written": summary.chunks_written,
        "output_files": [item.output_file.name for item in summary.prepared_files],
    }
    if len(summary.prepared_files) == 1:
        payload["output_file"] = summary.prepared_files[0].output_file.name

    with summary.summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _document_from_row(row: dict) -> LegalDocument:
    raw_date = row.get("date_decided")
    parsed_date = None
    if raw_date:
        try:
            parsed_date = date.fromisoformat(str(raw_date))
        except ValueError:
            parsed_date = None

    return LegalDocument(
        id=row["id"],
        doc_type=row["doc_type"],
        full_text=row["full_text"],
        case_name=row.get("case_name"),
        citation=row.get("citation"),
        court=row.get("court"),
        court_level=row.get("court_level"),
        date_decided=parsed_date,
        title=row.get("title"),
        section=row.get("section"),
        source_file=row.get("source_file"),
        is_privileged=bool(row.get("is_privileged", False)),
    )


def _output_name_for(raw_filename: str) -> str:
    if raw_filename.endswith("_cases.jsonl"):
        return raw_filename.replace("_cases.jsonl", "_chunks.jsonl")
    if raw_filename.endswith(".jsonl"):
        return raw_filename[:-6] + "_chunks.jsonl"
    return raw_filename + "_chunks.jsonl"


__all__ = [
    "PreparedFile",
    "PreparationSummary",
    "find_raw_files",
    "load_raw_documents",
    "prepare_corpus",
    "write_preparation_summary",
    "write_processed_chunks",
]
