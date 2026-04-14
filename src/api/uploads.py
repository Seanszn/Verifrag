"""Upload routes for server-ingested user documents."""

from __future__ import annotations

import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from src.api.dependencies import get_current_user
from src.config import DATA_DIR
from src.ingestion.chunker import chunk_document
from src.ingestion.document import LegalDocument
from src.ingestion.pdf_parser import PDFParserError, parse_pdf_to_document

try:
    from docx import Document as DocxDocument
except ImportError:  # pragma: no cover - depends on optional runtime package
    DocxDocument = None


router = APIRouter(tags=["uploads"])

USER_UPLOADS_ROOT = DATA_DIR / "uploads"
ALLOWED_UPLOAD_SUFFIXES = {".pdf", ".txt", ".md", ".docx"}
MAX_UPLOADS_PER_REQUEST = 10
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
RAW_FILENAME = "user_uploads.jsonl"
PROCESSED_FILENAME = "user_upload_chunks.jsonl"


class UploadFileSummary(BaseModel):
    """Metadata for one uploaded file."""

    filename: str
    content_type: str | None
    document_id: str
    size_bytes: int
    chunk_count: int
    is_privileged: bool


class UploadResponse(BaseModel):
    """Server response for upload ingestion."""

    conversation_id: int | None
    files_uploaded: int
    documents_upserted: int
    chunks_upserted: int
    files: list[UploadFileSummary]


class UnsupportedUploadError(ValueError):
    """Raised when an uploaded file cannot be processed."""


@router.post(
    "/api/uploads",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_documents(
    files: list[UploadFile] = File(...),
    conversation_id: int | None = Form(default=None),
    is_privileged: bool = Form(default=True),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> UploadResponse:
    """Receive user files, parse them into documents, and persist chunk JSONL."""
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files were uploaded.")
    if len(files) > MAX_UPLOADS_PER_REQUEST:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Upload at most {MAX_UPLOADS_PER_REQUEST} files per request.",
        )

    parsed_uploads: list[dict[str, Any]] = []
    for uploaded_file in files:
        filename = _sanitize_filename(uploaded_file.filename)
        try:
            file_bytes = await uploaded_file.read()
            _validate_upload(filename, file_bytes)
            document = _document_from_upload(file_bytes, filename, is_privileged=is_privileged)
            chunks = chunk_document(document)
        except UnsupportedUploadError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except PDFParserError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        finally:
            await uploaded_file.close()

        parsed_uploads.append(
            {
                "filename": filename,
                "content_type": uploaded_file.content_type,
                "size_bytes": len(file_bytes),
                "document": document,
                "chunks": chunks,
                "file_bytes": file_bytes,
            }
        )

    user_root = USER_UPLOADS_ROOT / f"user_{current_user['id']}"
    raw_path = user_root / "raw" / RAW_FILENAME
    processed_path = user_root / "processed" / PROCESSED_FILENAME
    stored_files_dir = user_root / "files"

    raw_rows = [_document_to_row(item["document"]) for item in parsed_uploads]
    processed_rows = [chunk.to_dict() for item in parsed_uploads for chunk in item["chunks"]]

    for item in parsed_uploads:
        _store_original_file(
            stored_files_dir=stored_files_dir,
            filename=item["filename"],
            document_id=item["document"].id,
            file_bytes=item["file_bytes"],
        )

    _upsert_jsonl_rows(raw_path, raw_rows)
    _upsert_jsonl_rows(processed_path, processed_rows)

    return UploadResponse(
        conversation_id=conversation_id,
        files_uploaded=len(parsed_uploads),
        documents_upserted=len(raw_rows),
        chunks_upserted=len(processed_rows),
        files=[
            UploadFileSummary(
                filename=item["filename"],
                content_type=item["content_type"],
                document_id=item["document"].id,
                size_bytes=item["size_bytes"],
                chunk_count=len(item["chunks"]),
                is_privileged=item["document"].is_privileged,
            )
            for item in parsed_uploads
        ],
    )


def _validate_upload(filename: str, file_bytes: bytes) -> None:
    if not filename:
        raise UnsupportedUploadError("Uploaded files must have a filename.")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_UPLOAD_SUFFIXES))
        raise UnsupportedUploadError(f"Unsupported file type '{suffix or '<none>'}'. Allowed: {allowed}.")
    if not file_bytes:
        raise UnsupportedUploadError(f"Uploaded file '{filename}' is empty.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise UnsupportedUploadError(
            f"Uploaded file '{filename}' exceeds the {MAX_UPLOAD_BYTES} byte limit."
        )


def _document_from_upload(file_bytes: bytes, filename: str, *, is_privileged: bool) -> LegalDocument:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return parse_pdf_to_document(
            file_bytes,
            filename=filename,
            is_privileged=is_privileged,
        )
    if suffix in {".txt", ".md"}:
        return _text_document_from_bytes(file_bytes, filename, is_privileged=is_privileged)
    if suffix == ".docx":
        return _docx_document_from_bytes(file_bytes, filename, is_privileged=is_privileged)
    raise UnsupportedUploadError(f"Unsupported file type '{suffix}'.")


def _text_document_from_bytes(
    file_bytes: bytes,
    filename: str,
    *,
    is_privileged: bool,
) -> LegalDocument:
    text = _decode_text_bytes(file_bytes).strip()
    if not text:
        raise UnsupportedUploadError(f"Uploaded text file '{filename}' did not contain readable text.")
    return LegalDocument(
        id=_make_document_id(filename, text),
        doc_type="user_upload",
        full_text=text,
        case_name=Path(filename).stem.replace("_", " ").replace("-", " ").strip() or filename,
        source_file=filename,
        is_privileged=is_privileged,
    )


def _docx_document_from_bytes(
    file_bytes: bytes,
    filename: str,
    *,
    is_privileged: bool,
) -> LegalDocument:
    if DocxDocument is None:
        raise UnsupportedUploadError("DOCX uploads require the python-docx package.")

    try:
        doc = DocxDocument(io.BytesIO(file_bytes))
    except Exception as exc:  # pragma: no cover - parser-specific failure mode
        raise UnsupportedUploadError(f"Failed to parse DOCX file '{filename}'.") from exc

    paragraphs = [paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text.strip()]
    text = "\n\n".join(paragraphs).strip()
    if not text:
        raise UnsupportedUploadError(f"Uploaded DOCX file '{filename}' did not contain readable text.")

    return LegalDocument(
        id=_make_document_id(filename, text),
        doc_type="user_upload",
        full_text=text,
        case_name=Path(filename).stem.replace("_", " ").replace("-", " ").strip() or filename,
        source_file=filename,
        is_privileged=is_privileged,
    )


def _decode_text_bytes(file_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnsupportedUploadError("Text upload could not be decoded as UTF-8 or a common fallback encoding.")


def _store_original_file(
    *,
    stored_files_dir: Path,
    filename: str,
    document_id: str,
    file_bytes: bytes,
) -> Path:
    stored_files_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower()
    stored_path = stored_files_dir / f"{document_id}{suffix}"
    stored_path.write_bytes(file_bytes)
    return stored_path


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


def _sanitize_filename(filename: str | None) -> str:
    if not filename:
        return ""
    basename = Path(filename).name.strip()
    sanitized = re.sub(r"[^A-Za-z0-9._ -]+", "_", basename)
    return sanitized.strip(" .")


def _make_document_id(filename: str, text: str) -> str:
    stem = Path(filename).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-") or "upload"
    digest = hashlib.sha1(text[:4000].encode("utf-8")).hexdigest()[:10]
    return f"upload_{slug}_{digest}"
