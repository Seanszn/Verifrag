"""Ingestion module for downloading and processing legal documents."""

from src.ingestion.chunker import chunk_document
from src.ingestion.user_file_ingestion import (
    UnsupportedUserFileError,
    UserFileCorpusIngestor,
    UserFileIngestionSummary,
)

__all__ = [
    "chunk_document",
    "UnsupportedUserFileError",
    "UserFileCorpusIngestor",
    "UserFileIngestionSummary",
]
