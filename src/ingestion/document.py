"""
Data models for legal documents and chunks.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
from datetime import date
import numpy as np

@dataclass
class LegalDocument:
    """A legal document (case or statute) with metadata."""

    id: str
    doc_type: Literal["case", "statute", "regulation", "user_upload"]
    full_text: str

    # Case-specific
    case_name: Optional[str] = None
    citation: Optional[str] = None  # e.g., "384 U.S. 436"
    court: Optional[str] = None
    court_level: Optional[str] = None  # scotus, circuit, district, etc.
    date_decided: Optional[date] = None

    # Statute-specific
    title: Optional[int] = None
    section: Optional[str] = None

    # User upload specific
    source_file: Optional[str] = None
    is_privileged: bool = False


@dataclass
class LegalChunk:
    """A chunk of a legal document for indexing."""

    id: str
    doc_id: str
    text: str
    chunk_index: int

    # Inherited metadata
    doc_type: str
    case_name: Optional[str] = None
    court: Optional[str] = None
    court_level: Optional[str] = None
    citation: Optional[str] = None
    date_decided: Optional[date] = None
    title: Optional[int] = None
    section: Optional[str] = None
    source_file: Optional[str] = None

    # Embedding (populated during indexing)
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "doc_type": self.doc_type,
            "case_name": self.case_name,
            "court": self.court,
            "court_level": self.court_level,
            "citation": self.citation,
            "date_decided": str(self.date_decided) if self.date_decided else None,
            "title": self.title,
            "section": self.section,
            "source_file": self.source_file,
        }
