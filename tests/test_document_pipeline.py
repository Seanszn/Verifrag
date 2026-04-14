# tests/test_document_pipeline.py

import pytest
from pathlib import Path
import json

from src.ingestion.chunker import Chunker, LegalChunk
from src.ingestion.pdf_parser import parse_pdf_to_document, PDFParserError

# Paths
SAMPLE_PDF_PATH = Path(__file__).parent / "sample.pdf"
CHUNKS_JSON_PATH = Path(__file__).parent / "sample_chunks.json"

#@pytest.mark.pipeline
def test_parser_and_chunker_pipeline():

    # Ensure PDF exists
    assert SAMPLE_PDF_PATH.exists(), f"Sample PDF not found at {SAMPLE_PDF_PATH}"

    # Parse PDF
    try:
        legal_doc = parse_pdf_to_document(SAMPLE_PDF_PATH)
    except PDFParserError as exc:
        pytest.fail(f"PDF parsing failed: {exc}")

    assert legal_doc.full_text, "PDF text extraction returned empty text"

    # Chunk the document
    chunker = Chunker()
    chunks = chunker.chunk_document(legal_doc)

    assert chunks, "Chunker returned no chunks"
    assert all(isinstance(c, LegalChunk) for c in chunks), "Chunks are not LegalChunk objects"

    # Convert chunks to dict format for JSON
    chunks_data = []
    for c in chunks:
        chunk_dict = {
            "id": c.id,
            "doc_id": c.doc_id,
            "text": c.text,
            "chunk_index": c.chunk_index,
            "doc_type": getattr(c, "doc_type", "pdf"),
            "court_level": getattr(c, "court_level", None),
            "citation": getattr(c, "citation", None),
            "date_decided": getattr(c, "date_decided", None),
            "embedding": getattr(c, "embedding", None),
            "section_type": getattr(c, "section_type", None),
        }
        chunks_data.append(chunk_dict)

    # Write JSON to file
    with CHUNKS_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    print(f"SUCCESS: Parsed '{SAMPLE_PDF_PATH.name}' into {len(chunks)} chunks")
    print(f"Chunks saved to {CHUNKS_JSON_PATH}")