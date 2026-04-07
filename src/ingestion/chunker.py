"""
Legal Document Chunker.
Splits parsed documents into overlapping LegalChunks based on paragraph boundaries.
"""

import re
from typing import List
from src.ingestion.document import LegalDocument, LegalChunk

class Chunker:
    def __init__(self, chunk_size: int = 1500, overlap: int = 250):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, doc: LegalDocument) -> List[LegalChunk]:
        """Split a document's full text into overlapping chunks."""
        # Split by double newline to respect paragraphs
        paragraphs = re.split(r'\n\n+', doc.full_text)
        
        chunks = []
        current_text = ""
        chunk_index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph pushes us over the size limit, save the chunk
            if len(current_text) + len(para) > self.chunk_size and current_text:
                chunks.append(self._create_chunk(doc, current_text, chunk_index))
                chunk_index += 1
                
                # Start the next chunk with overlapping text from the end of the previous one
                overlap_text = current_text[-self.overlap:]
                
                # Try to snap to the nearest space so we don't cut a word in half
                space_idx = overlap_text.find(" ")
                if space_idx != -1:
                    overlap_text = overlap_text[space_idx + 1:]
                
                current_text = overlap_text + "\n\n" + para
            else:
                if current_text:
                    current_text += "\n\n" + para
                else:
                    current_text = para

        # Don't forget the final chunk
        if current_text:
            chunks.append(self._create_chunk(doc, current_text, chunk_index))

        return chunks

    def _create_chunk(self, doc: LegalDocument, text: str, index: int) -> LegalChunk:
        return LegalChunk(
            id=f"{doc.id}_chunk_{index}",
            doc_id=doc.id,
            text=text.strip(),
            chunk_index=index,
            doc_type=doc.doc_type,
            court_level=doc.court_level,
            citation=doc.citation,
            date_decided=doc.date_decided
        )