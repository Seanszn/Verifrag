"""Legal document chunking."""

from __future__ import annotations

import re

from src.config import RETRIEVAL
from src.ingestion.document import LegalChunk, LegalDocument


_WHITESPACE_RE = re.compile(r"\s+")


def chunk_document(
    document: LegalDocument,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[LegalChunk]:
    """Split a legal document into overlapping retrieval chunks."""
    size = int(chunk_size or RETRIEVAL.chunk_size)
    overlap = int(chunk_overlap or RETRIEVAL.chunk_overlap)

    if size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = _tokenize_words(_normalize_text(document.full_text))
    if not words:
        return []

    chunks: list[LegalChunk] = []
    for chunk_index, text in enumerate(_window_words(words, size=size, overlap=overlap)):
        chunks.append(
            LegalChunk(
                id=f"{document.id}:{chunk_index}",
                doc_id=document.id,
                text=text,
                chunk_index=chunk_index,
                doc_type=document.doc_type,
                case_name=document.case_name,
                court=document.court,
                court_level=document.court_level,
                citation=document.citation,
                date_decided=document.date_decided,
                title=document.title,
                section=document.section,
                source_file=document.source_file,
            )
        )

    return chunks


class Chunker:
    """Compatibility wrapper for the original paragraph/character chunker API."""

    def __init__(self, chunk_size: int = 1500, overlap: int = 250) -> None:
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)

    def chunk_document(self, doc: LegalDocument) -> list[LegalChunk]:
        paragraphs = re.split(r"\n\n+", doc.full_text)
        chunks: list[LegalChunk] = []
        current_text = ""
        chunk_index = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(current_text) + len(paragraph) > self.chunk_size and current_text:
                chunks.append(self._create_chunk(doc, current_text, chunk_index))
                chunk_index += 1
                current_text = self._overlap_text(current_text)
                current_text = f"{current_text}\n\n{paragraph}" if current_text else paragraph
            else:
                current_text = f"{current_text}\n\n{paragraph}" if current_text else paragraph

        if current_text:
            chunks.append(self._create_chunk(doc, current_text, chunk_index))

        return chunks

    def _overlap_text(self, text: str) -> str:
        if self.overlap <= 0:
            return ""
        overlap_text = text[-self.overlap:]
        space_idx = overlap_text.find(" ")
        if space_idx != -1:
            overlap_text = overlap_text[space_idx + 1:]
        return overlap_text

    @staticmethod
    def _create_chunk(doc: LegalDocument, text: str, index: int) -> LegalChunk:
        return LegalChunk(
            id=f"{doc.id}_chunk_{index}",
            doc_id=doc.id,
            text=text.strip(),
            chunk_index=index,
            doc_type=doc.doc_type,
            case_name=doc.case_name,
            court=doc.court,
            court_level=doc.court_level,
            citation=doc.citation,
            date_decided=doc.date_decided,
            title=doc.title,
            section=doc.section,
            source_file=doc.source_file,
        )


def _normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return _WHITESPACE_RE.sub(" ", normalized)


def _tokenize_words(text: str) -> list[str]:
    return [token for token in text.split(" ") if token]


def _window_words(words: list[str], *, size: int, overlap: int) -> list[str]:
    step = size - overlap
    windows: list[str] = []
    for start in range(0, len(words), step):
        chunk_words = words[start:start + size]
        if not chunk_words:
            break
        windows.append(" ".join(chunk_words))
        if start + size >= len(words):
            break
    return windows


__all__ = ["Chunker", "chunk_document"]
