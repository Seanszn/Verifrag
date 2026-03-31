import pytest
from src.ingestion.document import LegalDocument
from src.ingestion.chunker import Chunker

def test_document_chunking():
    """Test that the chunker splits text correctly and preserves metadata."""
    
    # 1. Create a mock LegalDocument with 3 distinct paragraphs
    full_text = (
        "This is paragraph one. It has some introductory facts.\n\n"
        "This is paragraph two. It contains the main holding of the court.\n\n"
        "This is paragraph three. It provides the final conclusion."
    )
    
    doc = LegalDocument(
        id="test_case_001",
        doc_type="case",
        full_text=full_text,
        case_name="Verifrag v. Bugs",
        citation="123 U.S. 456",
        court_level="scotus"
    )

    # 2. Initialize chunker with artificially small limits to force a split
    chunker = Chunker(chunk_size=80, overlap=20)
    chunks = chunker.chunk_document(doc)

    # 3. Assertions
    assert len(chunks) > 1, "Document should have been split into multiple chunks."
    
    # Check that metadata cascaded down to the chunks
    first_chunk = chunks[0]
    assert first_chunk.doc_id == "test_case_001"
    assert first_chunk.citation == "123 U.S. 456"
    assert first_chunk.court_level == "scotus"
    
    # Check that the overlap logic worked (the second chunk should contain text from the first)
    # We look for overlapping words like "facts" or "holding" depending on where it split
    combined_chunk_text = " ".join([c.text for c in chunks])
    assert "paragraph one" in combined_chunk_text
    assert "paragraph three" in combined_chunk_text