"""Shared helpers for user-supplied document ingestion."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.ingestion.document import LegalDocument


TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


class UserDocumentError(ValueError):
    """Raised when a user-supplied document cannot be normalized."""


def decode_text_bytes(file_bytes: bytes, *, error_message: str | None = None) -> str:
    """Decode user-provided text with common encodings used by legal files."""
    for encoding in TEXT_ENCODINGS:
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UserDocumentError(error_message or "User document could not be decoded as text.")


def make_user_document_id(filename: str, text: str) -> str:
    """Create a stable upload document id from the filename and text prefix."""
    stem = Path(filename).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-") or "upload"
    digest = hashlib.sha1(text[:4000].encode("utf-8")).hexdigest()[:10]
    return f"upload_{slug}_{digest}"


def user_document_to_row(document: LegalDocument) -> dict[str, Any]:
    """Serialize a user upload document for raw JSONL storage."""
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


def upsert_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows by id while preserving existing row order where possible."""
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
