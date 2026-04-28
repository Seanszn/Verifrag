"""Tests for legal-aware chunking."""

from __future__ import annotations

from src.ingestion.chunker import Chunker, chunk_document
from src.ingestion.document import LegalDocument


def test_chunk_document_preserves_overlap_and_metadata():
    text = " ".join(f"word{i}" for i in range(12))
    document = LegalDocument(
        id="upload_alpha",
        doc_type="user_upload",
        full_text=text,
        case_name="Upload Alpha",
        court="scotus",
        citation="123 U.S. 456",
        court_level="scotus",
        source_file="upload_alpha.txt",
    )

    chunks = chunk_document(document, chunk_size=5, chunk_overlap=2)

    assert [chunk.id for chunk in chunks] == [
        "upload_alpha:0",
        "upload_alpha:1",
        "upload_alpha:2",
        "upload_alpha:3",
    ]
    assert chunks[0].text == "word0 word1 word2 word3 word4"
    assert chunks[1].text == "word3 word4 word5 word6 word7"
    assert all(chunk.doc_id == "upload_alpha" for chunk in chunks)
    assert all(chunk.case_name == "Upload Alpha" for chunk in chunks)
    assert all(chunk.court == "scotus" for chunk in chunks)
    assert all(chunk.citation == "123 U.S. 456" for chunk in chunks)
    assert all(chunk.court_level == "scotus" for chunk in chunks)
    assert all(chunk.source_file == "upload_alpha.txt" for chunk in chunks)


def test_chunk_document_handles_headings_without_changing_chunk_contract():
    document = LegalDocument(
        id="upload_beta",
        doc_type="user_upload",
        full_text=(
            "FACTS\n\n"
            "The officer entered the home without a warrant.\n\n"
            "ANALYSIS\n\n"
            "The exclusionary rule applies to the search."
        ),
    )

    chunks = chunk_document(document, chunk_size=7, chunk_overlap=1)

    assert chunks
    assert chunks[0].id == "upload_beta:0"
    assert all(chunk.doc_id == "upload_beta" for chunk in chunks)


def test_legacy_chunker_class_splits_paragraphs_and_preserves_metadata():
    full_text = (
        "This is paragraph one. It has some introductory facts.\n\n"
        "This is paragraph two. It contains the main holding of the court.\n\n"
        "This is paragraph three. It provides the final conclusion."
    )
    document = LegalDocument(
        id="test_case_001",
        doc_type="case",
        full_text=full_text,
        case_name="Verifrag v. Bugs",
        citation="123 U.S. 456",
        court_level="scotus",
    )

    chunks = Chunker(chunk_size=80, overlap=20).chunk_document(document)

    assert len(chunks) > 1
    assert chunks[0].doc_id == "test_case_001"
    assert chunks[0].case_name == "Verifrag v. Bugs"
    assert chunks[0].citation == "123 U.S. 456"
    assert chunks[0].court_level == "scotus"
    combined_chunk_text = " ".join(chunk.text for chunk in chunks)
    assert "paragraph one" in combined_chunk_text
    assert "paragraph three" in combined_chunk_text
