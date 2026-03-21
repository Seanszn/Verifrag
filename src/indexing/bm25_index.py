"""
BM25 sparse index.
"""

from __future__ import annotations

import pickle
import re
from collections import Counter
from datetime import date
import math
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    class BM25Okapi:  # type: ignore[no-redef]
        """Lightweight fallback BM25 implementation used when rank_bm25 is absent."""

        def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
            self.corpus = [list(doc) for doc in corpus]
            self.k1 = k1
            self.b = b
            self.doc_count = len(self.corpus)
            self.doc_lengths = [len(doc) for doc in self.corpus]
            self.avgdl = sum(self.doc_lengths) / self.doc_count if self.doc_count else 0.0
            self.term_freqs = [Counter(doc) for doc in self.corpus]

            doc_freqs = Counter()
            for doc in self.corpus:
                doc_freqs.update(set(doc))

            self.idf = {
                term: math.log(1.0 + (self.doc_count - freq + 0.5) / (freq + 0.5))
                for term, freq in doc_freqs.items()
            }

        def get_scores(self, query_tokens: Sequence[str]) -> List[float]:
            scores: List[float] = []
            for freqs, doc_length in zip(self.term_freqs, self.doc_lengths):
                norm = self.k1 * (1.0 - self.b + self.b * (doc_length / self.avgdl if self.avgdl else 0.0))
                score = 0.0
                for token in query_tokens:
                    term_freq = freqs.get(token, 0)
                    if term_freq == 0:
                        continue
                    idf = self.idf.get(token, 0.0)
                    numerator = term_freq * (self.k1 + 1.0)
                    denominator = term_freq + norm
                    score += idf * (numerator / denominator)
                scores.append(score)
            return scores

from src.config import INDEX_DIR
from src.ingestion.document import LegalChunk


def _default_tokenizer(text: str) -> List[str]:
    """Tokenize text into lowercase word terms for sparse retrieval."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Index:
    """Simple local BM25 index over ``LegalChunk`` instances."""

    def __init__(
        self,
        chunks: Sequence[LegalChunk] | None = None,
        *,
        tokenizer: Callable[[str], List[str]] | None = None,
        index_path: Path | None = None,
    ) -> None:
        self.tokenizer = tokenizer or _default_tokenizer
        self.index_path = index_path or (INDEX_DIR / "bm25.pkl")
        self._chunks: List[LegalChunk] = []
        self._tokenized_corpus: List[List[str]] = []
        self._bm25: BM25Okapi | None = None

        if chunks:
            self.build(chunks)

    def build(self, chunks: Sequence[LegalChunk]) -> None:
        """Replace the current corpus and rebuild the sparse index."""
        self._chunks = list(chunks)
        self._tokenized_corpus = [self.tokenizer(chunk.text) for chunk in self._chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None

    def add(self, chunks: Sequence[LegalChunk] | Iterable[LegalChunk]) -> None:
        """Append chunks and rebuild the BM25 model."""
        incoming = list(chunks)
        if not incoming:
            return
        self.build([*self._chunks, *incoming])

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float, dict]]:
        """
        Search the corpus using BM25.

        Returns items in the same shape as the vector store:
        ``(chunk_id, score, metadata)``.
        """
        if k <= 0 or not query or self._bm25 is None:
            return []

        query_tokens = self.tokenizer(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: float(scores[idx]),
            reverse=True,
        )

        results: List[Tuple[str, float, dict]] = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            chunk = self._chunks[idx]
            results.append((chunk.id, score, chunk.to_dict()))
            if len(results) >= k:
                break

        return results

    def count(self) -> int:
        """Return number of indexed chunks."""
        return len(self._chunks)

    def save(self) -> None:
        """Persist the chunk corpus needed to rebuild the BM25 index."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": [chunk.to_dict() for chunk in self._chunks],
        }
        with self.index_path.open("wb") as fh:
            pickle.dump(payload, fh)

    def load(self) -> None:
        """Load persisted chunks and rebuild the BM25 index."""
        with self.index_path.open("rb") as fh:
            payload = pickle.load(fh)

        chunks = [self._chunk_from_dict(item) for item in payload.get("chunks", [])]
        self.build(chunks)

    @staticmethod
    def _chunk_from_dict(data: dict) -> LegalChunk:
        raw_date = data.get("date_decided")
        parsed_date = date.fromisoformat(raw_date) if raw_date else None
        return LegalChunk(
            id=data["id"],
            doc_id=data["doc_id"],
            text=data["text"],
            chunk_index=data["chunk_index"],
            doc_type=data["doc_type"],
            court_level=data.get("court_level"),
            citation=data.get("citation"),
            date_decided=parsed_date,
            section_type=data.get("section_type"),
        )
