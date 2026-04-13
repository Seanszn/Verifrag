"""Local user-file ingestion into raw and processed JSONL corpora."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.config import PROCESSED_DIR, RAW_DIR
from src.ingestion.chunker import chunk_document
from src.ingestion.document import LegalDocument


RAW_FILENAME = "user_uploads.jsonl"
PROCESSED_FILENAME = "user_upload_chunks.jsonl"
SUPPORTED_SUFFIXES = {".txt", ".md"}


class UnsupportedUserFileError(ValueError):
    """Raised when an explicit user file cannot be ingested."""


@dataclass(frozen=True)
class UserFileIngestionSummary:
    files_discovered: int
    files_ingested: int
    documents_upserted: int
    chunks_upserted: int
    document_ids: list[str]


class UserFileCorpusIngestor:
    """Ingest local user text/markdown files into corpus JSONL files."""

    def __init__(
        self,
        *,
        raw_dir: str | Path = RAW_DIR,
        processed_dir: str | Path = PROCESSED_DIR,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

    def ingest_path(self, path: str | Path, *, is_privileged: bool = False) -> UserFileIngestionSummary:
        return self.ingest_paths([path], is_privileged=is_privileged, reject_unsupported_files=True)

    def ingest_paths(
        self,
        paths: Sequence[str | Path] | Iterable[str | Path],
        *,
        is_privileged: bool = False,
        reject_unsupported_files: bool = False,
    ) -> UserFileIngestionSummary:
        files = self._discover_files(paths, reject_unsupported_files=reject_unsupported_files)
        documents = [self._document_from_file(path, is_privileged=is_privileged) for path in files]
        raw_rows = [self._document_to_row(document) for document in documents]
        processed_rows = [
            chunk.to_dict()
            for document in documents
            for chunk in chunk_document(document)
        ]

        if raw_rows:
            _upsert_jsonl_rows(self.raw_dir / RAW_FILENAME, raw_rows)
        if processed_rows:
            _upsert_jsonl_rows(self.processed_dir / PROCESSED_FILENAME, processed_rows)

        return UserFileIngestionSummary(
            files_discovered=len(files),
            files_ingested=len(documents),
            documents_upserted=len(raw_rows),
            chunks_upserted=len(processed_rows),
            document_ids=[document.id for document in documents],
        )

    def _discover_files(
        self,
        paths: Sequence[str | Path] | Iterable[str | Path],
        *,
        reject_unsupported_files: bool,
    ) -> list[Path]:
        discovered: list[Path] = []
        for raw_path in paths:
            path = Path(raw_path)
            if path.is_dir():
                discovered.extend(
                    child
                    for child in sorted(path.rglob("*"))
                    if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES
                )
                continue
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                discovered.append(path)
                continue
            if reject_unsupported_files:
                suffix = path.suffix.lower() or "<none>"
                raise UnsupportedUserFileError(f"Unsupported user file type '{suffix}'.")
        return discovered

    def _document_from_file(self, path: Path, *, is_privileged: bool) -> LegalDocument:
        text = _decode_text(path.read_bytes()).strip()
        if not text:
            raise UnsupportedUserFileError(f"User file '{path.name}' did not contain readable text.")

        return LegalDocument(
            id=_make_document_id(path.name, text),
            doc_type="user_upload",
            full_text=text,
            case_name=path.stem.replace("_", " ").replace("-", " ").strip() or path.name,
            source_file=path.name,
            is_privileged=is_privileged,
        )

    @staticmethod
    def _document_to_row(document: LegalDocument) -> dict[str, Any]:
        return {
            "id": document.id,
            "doc_type": document.doc_type,
            "full_text": document.full_text,
            "case_name": document.case_name,
            "citation": document.citation,
            "court": document.court,
            "court_level": document.court_level,
            "date_decided": document.date_decided.isoformat() if document.date_decided else None,
            "source_file": document.source_file,
            "is_privileged": document.is_privileged,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }


def _decode_text(file_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnsupportedUserFileError("User file could not be decoded as text.")


def _upsert_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_ids: list[str] = []
    rows_by_id: dict[str, dict[str, Any]] = {}

    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                payload = line.strip()
                if not payload:
                    continue
                try:
                    row = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid JSON in {path} at line {line_number}.") from exc
                row_id = str(row["id"])
                if row_id not in rows_by_id:
                    ordered_ids.append(row_id)
                rows_by_id[row_id] = row

    for row in rows:
        row_id = str(row["id"])
        if row_id not in rows_by_id:
            ordered_ids.append(row_id)
        rows_by_id[row_id] = row

    with path.open("w", encoding="utf-8") as handle:
        for row_id in ordered_ids:
            handle.write(json.dumps(rows_by_id[row_id], ensure_ascii=False) + "\n")


def _make_document_id(filename: str, text: str) -> str:
    stem = Path(filename).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-") or "upload"
    digest = hashlib.sha1(text[:4000].encode("utf-8")).hexdigest()[:10]
    return f"upload_{slug}_{digest}"


__all__ = [
    "PROCESSED_FILENAME",
    "RAW_FILENAME",
    "SUPPORTED_SUFFIXES",
    "UnsupportedUserFileError",
    "UserFileCorpusIngestor",
    "UserFileIngestionSummary",
]
