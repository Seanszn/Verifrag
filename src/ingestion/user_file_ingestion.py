"""Local user-file ingestion into raw and processed JSONL corpora."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from src.config import PROCESSED_DIR, RAW_DIR
from src.ingestion.chunker import chunk_document
from src.ingestion.document import LegalDocument
from src.ingestion.user_documents import (
    UserDocumentError,
    decode_text_bytes,
    make_user_document_id,
    upsert_jsonl_rows,
    user_document_to_row,
)


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
        raw_rows = [user_document_to_row(document) for document in documents]
        processed_rows = [
            chunk.to_dict()
            for document in documents
            for chunk in chunk_document(document)
        ]

        if raw_rows:
            upsert_jsonl_rows(self.raw_dir / RAW_FILENAME, raw_rows)
        if processed_rows:
            upsert_jsonl_rows(self.processed_dir / PROCESSED_FILENAME, processed_rows)

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
        try:
            text = decode_text_bytes(
                path.read_bytes(),
                error_message="User file could not be decoded as text.",
            ).strip()
        except UserDocumentError as exc:
            raise UnsupportedUserFileError(str(exc)) from exc
        if not text:
            raise UnsupportedUserFileError(f"User file '{path.name}' did not contain readable text.")

        return LegalDocument(
            id=make_user_document_id(path.name, text),
            doc_type="user_upload",
            full_text=text,
            case_name=path.stem.replace("_", " ").replace("-", " ").strip() or path.name,
            source_file=path.name,
            is_privileged=is_privileged,
        )

__all__ = [
    "PROCESSED_FILENAME",
    "RAW_FILENAME",
    "SUPPORTED_SUFFIXES",
    "UnsupportedUserFileError",
    "UserFileCorpusIngestor",
    "UserFileIngestionSummary",
]
