"""
PDF parser for user-uploaded legal documents.

The parser is intentionally local-first and conservative:
- uses PyMuPDF for extraction
- removes common PDF layout artifacts
- preserves section-style line breaks that help downstream claim extraction
- returns LegalDocument instances with stable IDs and metadata
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Optional

from src.ingestion.document import LegalDocument

try:
    import fitz
except ImportError:  # pragma: no cover - exercised through runtime guard
    fitz = None


_MULTISPACE_RE = re.compile(r"[ \t]+")
_SOFT_HYPHEN_RE = re.compile(r"\u00ad")
_HEADER_FOOTER_RE = re.compile(
    r"^(page \d+(\s+of\s+\d+)?|\d+\s*$|filed\s+\d{1,2}/\d{1,2}/\d{2,4}.*)$",
    re.IGNORECASE,
)
_DOCKET_RE = re.compile(r"\b(no\.?|case\s+no\.?|docket\s+no\.?)\s+[\w:\-./]+\b", re.IGNORECASE)
_CITATION_RE = re.compile(r"\b\d{1,4}\s+[A-Z][A-Za-z.\d]*\s+\d{1,5}\b")
_DATE_RE = re.compile(
    r"\b("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    r")\s+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE,
)
_UPPERCASE_HEADING_RE = re.compile(r"^[A-Z][A-Z0-9 ,.'&()/-]{4,}$")
_SECTION_HEADING_RE = re.compile(
    r"^((section|article|chapter|part|count|claim|background|facts|analysis|discussion|conclusion)\b|[IVXLC]+\.)",
    re.IGNORECASE,
)
_STATUTE_LINE_RE = re.compile(
    r"^(\d+\s+U\.?\s*S\.?\s*C\.?|\d+\s+C\.?\s*F\.?\s*R\.?|§+\s*\d+)",
    re.IGNORECASE,
)


class PDFParserError(RuntimeError):
    """Raised when a PDF cannot be parsed into useful text."""


@dataclass(frozen=True)
class ParsedPDF:
    """Intermediate PDF parse result with layout-aware metadata."""

    text: str
    title: Optional[str]
    page_count: int
    source_file: Optional[str]
    metadata: dict[str, Any]

    def to_legal_document(
        self,
        *,
        document_id: Optional[str] = None,
        is_privileged: bool = False,
    ) -> LegalDocument:
        """Convert the parsed PDF into the ingestion document model."""
        source_name = Path(self.source_file).name if self.source_file else None
        resolved_id = document_id or _make_document_id(source_name, self.text)
        return LegalDocument(
            id=resolved_id,
            doc_type="user_upload",
            full_text=self.text,
            case_name=self.title,
            source_file=source_name,
            is_privileged=is_privileged,
        )


class PDFParser:
    """Parse uploaded PDFs into clean legal text."""

    def __init__(
        self,
        *,
        min_text_length: int = 80,
        join_short_lines: bool = True,
    ):
        self.min_text_length = min_text_length
        self.join_short_lines = join_short_lines

    def parse(
        self,
        source: str | Path | bytes | bytearray | BinaryIO,
        *,
        filename: Optional[str] = None,
        document_id: Optional[str] = None,
        is_privileged: bool = False,
    ) -> LegalDocument:
        """Parse a PDF from path, bytes, or file-like object into LegalDocument."""
        parsed = self.parse_pdf(source, filename=filename)
        return parsed.to_legal_document(
            document_id=document_id,
            is_privileged=is_privileged,
        )

    def parse_pdf(
        self,
        source: str | Path | bytes | bytearray | BinaryIO,
        *,
        filename: Optional[str] = None,
    ) -> ParsedPDF:
        """Extract and normalize PDF text, preserving claim-friendly structure."""
        _require_pymupdf()

        stream, source_name = self._coerce_source(source, filename=filename)
        try:
            document = fitz.open(stream=stream, filetype="pdf")
        except Exception as exc:  # pragma: no cover - fitz error types vary
            raise PDFParserError("Failed to open PDF.") from exc

        with document:
            if document.page_count == 0:
                raise PDFParserError("PDF has no pages.")

            page_texts = [self._extract_page_text(page) for page in document]
            combined_text = self._assemble_document_text(page_texts)

            if len(combined_text.strip()) < self.min_text_length:
                raise PDFParserError(
                    "PDF text extraction produced too little text. The file may be scanned or empty."
                )

            title = self._infer_title(document, page_texts, source_name)
            metadata = {
                "page_count": document.page_count,
                "title": title,
                "source_file": source_name,
                "parser": "pymupdf",
            }
            return ParsedPDF(
                text=combined_text,
                title=title,
                page_count=document.page_count,
                source_file=source_name,
                metadata=metadata,
            )

    def _coerce_source(
        self,
        source: str | Path | bytes | bytearray | BinaryIO,
        *,
        filename: Optional[str],
    ) -> tuple[bytes, Optional[str]]:
        if isinstance(source, (str, Path)):
            path = Path(source)
            return path.read_bytes(), filename or path.name

        if isinstance(source, (bytes, bytearray)):
            return bytes(source), filename

        if hasattr(source, "read"):
            data = source.read()
            if isinstance(data, str):
                data = data.encode("utf-8")
            if not isinstance(data, (bytes, bytearray)):
                raise PDFParserError("Uploaded file-like object did not return bytes.")
            resolved_name = filename or getattr(source, "name", None)
            return bytes(data), resolved_name

        raise TypeError("source must be a path, bytes, or a binary file-like object")

    def _extract_page_text(self, page: Any) -> str:
        # Block extraction gives better paragraph boundaries than plain text mode.
        blocks = page.get_text("blocks")
        ordered_blocks = sorted(blocks, key=lambda block: (round(block[1], 1), round(block[0], 1)))

        cleaned_blocks: list[str] = []
        for block in ordered_blocks:
            if len(block) < 5:
                continue
            text = str(block[4]).strip()
            if not text:
                continue
            normalized = _normalize_block(text)
            if normalized:
                cleaned_blocks.append(normalized)

        if not cleaned_blocks:
            plain = page.get_text("text")
            return _normalize_block(plain)

        return "\n\n".join(cleaned_blocks)

    def _assemble_document_text(self, page_texts: Iterable[str]) -> str:
        lines_by_page = [self._clean_page_lines(text) for text in page_texts if text.strip()]
        if not lines_by_page:
            return ""

        repeated_lines = _find_repeated_page_lines(lines_by_page)
        pages: list[str] = []
        for page_lines in lines_by_page:
            filtered = [
                line for line in page_lines
                if line not in repeated_lines and not _looks_like_page_artifact(line)
            ]
            page_text = self._merge_lines(filtered)
            if page_text:
                pages.append(page_text)

        text = "\n\n".join(page for page in pages if page.strip())
        return _finalize_text(text)

    def _clean_page_lines(self, text: str) -> list[str]:
        lines = []
        for raw_line in text.splitlines():
            line = _clean_line(raw_line)
            if not line:
                continue
            lines.append(line)
        return lines

    def _merge_lines(self, lines: list[str]) -> str:
        merged: list[str] = []
        current = ""

        for line in lines:
            if not current:
                current = line
                continue

            if _should_break_paragraph(current, line):
                merged.append(current)
                current = line
                continue

            if current.endswith("-") and _is_word_continuation(line):
                current = current[:-1] + line.lstrip()
                continue

            joiner = " " if self.join_short_lines else "\n"
            current = f"{current}{joiner}{line}".strip()

        if current:
            merged.append(current)

        return "\n\n".join(merged)

    def _infer_title(
        self,
        document: Any,
        page_texts: list[str],
        source_name: Optional[str],
    ) -> Optional[str]:
        metadata = document.metadata or {}
        for key in ("title", "subject"):
            value = metadata.get(key)
            if value and str(value).strip():
                return _normalize_title(str(value))

        first_page_lines = self._clean_page_lines(page_texts[0]) if page_texts else []
        for line in first_page_lines[:12]:
            if len(line) < 6:
                continue
            if _looks_like_page_artifact(line):
                continue
            if _looks_like_title(line):
                return _normalize_title(line)

        if source_name:
            return _normalize_title(Path(source_name).stem.replace("_", " ").replace("-", " "))
        return None


def parse_pdf_to_document(
    source: str | Path | bytes | bytearray | BinaryIO,
    *,
    filename: Optional[str] = None,
    document_id: Optional[str] = None,
    is_privileged: bool = False,
) -> LegalDocument:
    """Convenience helper for one-shot parsing."""
    return PDFParser().parse(
        source,
        filename=filename,
        document_id=document_id,
        is_privileged=is_privileged,
    )


def _require_pymupdf() -> None:
    if fitz is None:
        raise PDFParserError(
            "PyMuPDF is not installed. Install `pymupdf` to enable PDF parsing."
        )


def _normalize_block(text: str) -> str:
    text = _SOFT_HYPHEN_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = _MULTISPACE_RE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_line(line: str) -> str:
    line = _SOFT_HYPHEN_RE.sub("", line)
    line = line.replace("\t", " ")
    line = _MULTISPACE_RE.sub(" ", line).strip()
    line = line.strip(" |")
    return line


def _find_repeated_page_lines(lines_by_page: list[list[str]]) -> set[str]:
    counts: dict[str, int] = {}
    for lines in lines_by_page:
        unique = set()
        if lines:
            unique.add(lines[0])
            unique.add(lines[-1])
        for line in unique:
            counts[line] = counts.get(line, 0) + 1

    threshold = max(2, len(lines_by_page) // 2 + 1)
    return {
        line for line, count in counts.items()
        if count >= threshold and (_looks_like_page_artifact(line) or len(line) <= 80)
    }


def _looks_like_page_artifact(line: str) -> bool:
    normalized = line.strip()
    if not normalized:
        return True
    if _HEADER_FOOTER_RE.match(normalized):
        return True
    if normalized.startswith("\uf0b7"):
        return False
    if len(normalized) <= 4 and any(char.isdigit() for char in normalized):
        return True
    return False


def _should_break_paragraph(current: str, next_line: str) -> bool:
    if not current or not next_line:
        return True
    if current.endswith((".", ":", ";", "?", "!")):
        return True
    if _looks_like_heading(next_line):
        return True
    if next_line.startswith(("•", "-", "*")):
        return True
    if _STATUTE_LINE_RE.match(next_line):
        return True
    if re.match(r"^\(?[a-zA-Z0-9]{1,4}\)", next_line):
        return True
    return False


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) > 120:
        return False
    if _UPPERCASE_HEADING_RE.match(stripped):
        return True
    if _SECTION_HEADING_RE.match(stripped):
        return True
    return False


def _looks_like_title(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 6 or len(stripped) > 160:
        return False
    if _looks_like_heading(stripped):
        return True
    if " v. " in stripped or " vs. " in stripped:
        return True
    if stripped.istitle() and len(stripped.split()) >= 2:
        return True
    return False


def _is_word_continuation(line: str) -> bool:
    stripped = line.lstrip()
    return bool(stripped) and stripped[0].isalnum()


def _normalize_title(text: str) -> str:
    title = re.sub(r"\s+", " ", text).strip(" -_")
    if title.isupper():
        title = title.title()
    return title


def _make_document_id(source_name: Optional[str], text: str) -> str:
    base_name = Path(source_name).stem if source_name else "upload"
    slug = re.sub(r"[^a-z0-9]+", "-", base_name.lower()).strip("-") or "upload"
    digest = hashlib.sha1(text[:4000].encode("utf-8")).hexdigest()[:10]
    return f"upload_{slug}_{digest}"


def _finalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def summarize_for_claims(text: str, *, max_chars: int = 1200) -> str:
    """
    Produce a short parser-side synopsis for diagnostics or previews.

    This is not a semantic summary. It extracts claim-salient lines so callers can
    inspect whether the parser preserved holdings, sections, and citations.
    """
    lines = [_clean_line(line) for line in text.splitlines()]
    kept: list[str] = []
    for line in lines:
        if not line:
            continue
        if _looks_like_heading(line) or _DOCKET_RE.search(line) or _CITATION_RE.search(line) or _DATE_RE.search(line):
            kept.append(line)
        if sum(len(item) + 1 for item in kept) >= max_chars:
            break

    if not kept:
        snippet = _clean_line(text[:max_chars])
        return snippet[:max_chars]

    summary = "\n".join(kept)
    return summary[:max_chars].strip()


__all__ = [
    "PDFParser",
    "PDFParserError",
    "ParsedPDF",
    "parse_pdf_to_document",
    "summarize_for_claims",
]
