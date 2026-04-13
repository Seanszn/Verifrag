"""BM25 sparse index."""

from __future__ import annotations

import math
import pickle
import re
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

from src.ingestion.document import LegalChunk


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class BM25Index:
    """Simple BM25 index over legal chunk rows."""

    def __init__(
        self,
        chunks: Sequence[LegalChunk] | Sequence[str] | None = None,
        *,
        index_path: str | Path | None = None,
        save_path: str | Path | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.index_path = Path(index_path or save_path) if index_path or save_path else None
        self.k1 = float(k1)
        self.b = float(b)
        self._texts: list[str] = []
        self._metadatas: list[dict] = []
        self._result_ids: list[str] = []
        self._tokenized_corpus: list[list[str]] = []
        self._doc_lengths: list[int] = []
        self._avg_doc_length: float = 0.0
        self._idf: dict[str, float] = {}
        self._term_freqs: list[dict[str, int]] = []

        if chunks:
            self.build(chunks)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token.lower() for token in _TOKEN_RE.findall(text or "")]

    def build(
        self,
        chunks: Sequence[LegalChunk] | Iterable[LegalChunk] | Sequence[str],
        metadatas: Sequence[dict] | None = None,
    ) -> None:
        """Build from LegalChunk rows or the legacy texts/metadatas pair."""
        rows = list(chunks)
        if not rows:
            self._reset()
            return

        if isinstance(rows[0], LegalChunk):
            self._load_rows_from_chunks(rows)
        else:
            self._load_rows_from_texts([str(item) for item in rows], metadatas)

        self._tokenized_corpus = [self._tokenize(text) for text in self._texts]
        self._doc_lengths = [len(tokens) for tokens in self._tokenized_corpus]
        self._avg_doc_length = (
            sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0
        )
        self._term_freqs = []

        doc_freqs: dict[str, int] = {}
        for tokens in self._tokenized_corpus:
            term_freqs: dict[str, int] = {}
            for token in tokens:
                term_freqs[token] = term_freqs.get(token, 0) + 1
            self._term_freqs.append(term_freqs)
            for token in term_freqs:
                doc_freqs[token] = doc_freqs.get(token, 0) + 1

        total_docs = len(self._texts)
        self._idf = {
            token: math.log(1.0 + (total_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in doc_freqs.items()
        }

    def search(self, query: str, k: int = 5) -> list[tuple[str, float, dict]]:
        if k <= 0 or not self._texts:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        results: list[tuple[str, float, dict]] = []
        avgdl = self._avg_doc_length or 1.0

        for idx, (doc_len, term_freqs) in enumerate(zip(self._doc_lengths, self._term_freqs)):
            score = 0.0
            for term in query_terms:
                freq = term_freqs.get(term, 0)
                if freq <= 0:
                    continue
                idf = self._idf.get(term)
                if idf is None:
                    continue
                numerator = freq * (self.k1 + 1.0)
                denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / avgdl))
                score += idf * (numerator / denominator)

            if score > 0.0:
                results.append((self._result_ids[idx], float(score), dict(self._metadatas[idx])))

        results.sort(key=lambda item: (-item[1], item[0]))
        return results[:k]

    def save(self) -> None:
        if self.index_path is None:
            raise ValueError("index_path is required to save BM25Index")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": self._metadatas,
            "k1": self.k1,
            "b": self.b,
        }
        with self.index_path.open("wb") as handle:
            pickle.dump(payload, handle)

    def load(self) -> None:
        if self.index_path is None:
            raise ValueError("index_path is required to load BM25Index")

        with self.index_path.open("rb") as handle:
            payload = pickle.load(handle)

        self.k1 = float(payload.get("k1", 1.5))
        self.b = float(payload.get("b", 0.75))
        rows = [dict(row) for row in payload.get("chunks", [])]
        self._load_rows_from_dicts(rows)
        self.build([self._chunk_from_dict(row) for row in self._metadatas])

    def _reset(self) -> None:
        self._texts = []
        self._metadatas = []
        self._result_ids = []
        self._tokenized_corpus = []
        self._doc_lengths = []
        self._avg_doc_length = 0.0
        self._idf = {}
        self._term_freqs = []

    def _load_rows_from_chunks(self, chunks: Sequence[LegalChunk]) -> None:
        self._texts = [chunk.text for chunk in chunks]
        self._metadatas = [chunk.to_dict() for chunk in chunks]
        self._result_ids = [chunk.id for chunk in chunks]

    def _load_rows_from_texts(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict] | None,
    ) -> None:
        metadata_rows = list(metadatas or [{} for _ in texts])
        if len(metadata_rows) != len(texts):
            raise ValueError("texts and metadatas must have matching lengths")

        self._texts = list(texts)
        self._metadatas = []
        self._result_ids = []
        for idx, (text, metadata) in enumerate(zip(texts, metadata_rows)):
            payload = dict(metadata)
            chunk_id = str(payload.get("id", idx))
            payload["id"] = chunk_id
            payload.setdefault("doc_id", chunk_id)
            payload.setdefault("text", text)
            payload.setdefault("chunk_index", idx)
            payload.setdefault("doc_type", "case")
            self._metadatas.append(payload)
            self._result_ids.append(chunk_id)

    def _load_rows_from_dicts(self, rows: Sequence[dict]) -> None:
        self._load_rows_from_texts(
            [str(row.get("text", "")) for row in rows],
            [dict(row) for row in rows],
        )

    @staticmethod
    def _chunk_from_dict(payload: dict) -> LegalChunk:
        raw_date = payload.get("date_decided")
        parsed_date = None
        if raw_date:
            try:
                parsed_date = date.fromisoformat(str(raw_date))
            except ValueError:
                parsed_date = None

        chunk_id = str(payload["id"])
        return LegalChunk(
            id=chunk_id,
            doc_id=str(payload.get("doc_id", chunk_id)),
            text=str(payload.get("text", "")),
            chunk_index=int(payload.get("chunk_index", 0)),
            doc_type=str(payload.get("doc_type", "case")),
            case_name=payload.get("case_name"),
            court=payload.get("court"),
            court_level=payload.get("court_level"),
            citation=payload.get("citation"),
            date_decided=parsed_date,
            title=payload.get("title"),
            section=payload.get("section"),
            source_file=payload.get("source_file"),
        )
