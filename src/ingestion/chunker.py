"""
Legal-aware text chunking.
"""

from __future__ import annotations

import re
from collections import Counter

from src.config import RETRIEVAL
from src.ingestion.document import LegalChunk, LegalDocument

_BLANK_LINE_RE = re.compile(r"\n\s*\n+")
_MULTISPACE_RE = re.compile(r"[ \t]+")
_UPPERCASE_HEADING_RE = re.compile(r"^[A-Z][A-Z0-9 ,.'&()/-]{4,}$")
_SECTION_HEADING_RE = re.compile(
    r"^((section|article|chapter|part|count|claim|background|facts|analysis|discussion|"
    r"conclusion|holding|issue|reasoning)\b|[IVXLC]+\.)",
    re.IGNORECASE,
)
_SECTION_KEYWORDS = {
    "facts": "facts",
    "background": "facts",
    "procedural": "procedural",
    "procedure": "procedural",
    "holding": "holding",
    "analysis": "analysis",
    "discussion": "analysis",
    "reasoning": "analysis",
    "conclusion": "conclusion",
}


def chunk_document(
    document: LegalDocument,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[LegalChunk]:
    """Split a legal document into overlapping chunks for indexing."""
    text = _normalize_text(document.full_text)
    if not text:
        return []

    resolved_chunk_size = max(1, int(chunk_size or RETRIEVAL.chunk_size))
    resolved_overlap = max(
        0,
        min(int(chunk_overlap or RETRIEVAL.chunk_overlap), resolved_chunk_size - 1),
    )
    step = max(1, resolved_chunk_size - resolved_overlap)

    word_stream = _tokenize_with_sections(document)
    if not word_stream:
        return []

    chunks: list[LegalChunk] = []
    for chunk_index, start in enumerate(range(0, len(word_stream), step)):
        window = word_stream[start : start + resolved_chunk_size]
        if not window:
            continue

        chunk_text = " ".join(word for word, _ in window).strip()
        if not chunk_text:
            continue

        chunks.append(
            LegalChunk(
                id=f"{document.id}:{chunk_index}",
                doc_id=document.id,
                text=chunk_text,
                chunk_index=chunk_index,
                doc_type=document.doc_type,
                court_level=document.court_level,
                citation=document.citation,
                date_decided=document.date_decided,
                section_type=_majority_section(window, document.doc_type),
            )
        )

    return chunks


def _tokenize_with_sections(document: LegalDocument) -> list[tuple[str, str | None]]:
    paragraphs = _split_paragraphs(document.full_text)
    current_section = "statute_text" if document.doc_type in {"statute", "regulation"} else None
    tokens: list[tuple[str, str | None]] = []

    for paragraph in paragraphs:
        if _looks_like_heading(paragraph):
            current_section = _classify_section_heading(paragraph) or current_section

        for word in paragraph.split():
            tokens.append((word, current_section))

    return tokens


def _split_paragraphs(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", normalized)
    normalized = _MULTISPACE_RE.sub(" ", normalized)
    paragraphs = []

    for raw_paragraph in _BLANK_LINE_RE.split(normalized):
        paragraph = raw_paragraph.strip()
        if not paragraph:
            continue
        paragraph = re.sub(r"\s*\n\s*", " ", paragraph)
        paragraphs.append(paragraph)

    return paragraphs


def _normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _looks_like_heading(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) > 120:
        return False
    if _UPPERCASE_HEADING_RE.match(stripped):
        return True
    if _SECTION_HEADING_RE.match(stripped):
        return True
    return False


def _classify_section_heading(text: str) -> str | None:
    lowered = text.lower()
    for keyword, section_name in _SECTION_KEYWORDS.items():
        if keyword in lowered:
            return section_name
    return None


def _majority_section(
    window: list[tuple[str, str | None]],
    doc_type: str,
) -> str | None:
    counts = Counter(section for _, section in window if section)
    if counts:
        return counts.most_common(1)[0][0]
    if doc_type in {"statute", "regulation"}:
        return "statute_text"
    return None


__all__ = ["chunk_document"]
