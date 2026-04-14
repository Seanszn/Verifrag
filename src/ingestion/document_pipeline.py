"""
End-to-end document processing pipeline:
- parses uploaded PDF into LegalDocument
- chunks into LegalChunks
- saves results to disk
"""

from pathlib import Path
import json
from typing import List

from src.ingestion.pdf_parser import parse_pdf_to_document
from src.ingestion.chunker import Chunker
from src.ingestion.document import LegalChunk, LegalDocument


class DocumentPipeline:
    def __init__(
        self,
        upload_dir: str = "user_uploads",
        chunk_size: int = 1500,
        overlap: int = 250,
    ):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        self.chunker = Chunker(chunk_size=chunk_size, overlap=overlap)

    def process_upload(self, file, filename: str) -> List[LegalChunk]:
        """
        Full pipeline:
        upload → parse → chunk → save
        """

        # Step 1: Parse PDF → LegalDocument
        document: LegalDocument = parse_pdf_to_document(
            file,
            filename=filename
        )

        # Step 2: Chunk
        chunks: List[LegalChunk] = self.chunker.chunk_document(document)

        # Step 3: Save results
        self._save_document(document)
        self._save_chunks(document.id, chunks)

        return chunks

    def _save_document(self, document: LegalDocument):
        doc_path = self.upload_dir / f"{document.id}.txt"
        doc_path.write_text(document.full_text, encoding="utf-8")

    def _save_chunks(self, doc_id: str, chunks: List[LegalChunk]):
        chunk_path = self.upload_dir / f"{doc_id}_chunks.json"

        serializable_chunks = [
            {
                "id": c.id,
                "doc_id": c.doc_id,
                "text": c.text,
                "chunk_index": c.chunk_index,
                "doc_type": c.doc_type,
                "citation": c.citation,
                "court_level": c.court_level,
                "date_decided": str(c.date_decided) if c.date_decided else None,
            }
            for c in chunks
        ]

        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(serializable_chunks, f, indent=2)