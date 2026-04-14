"""Hybrid retrieval with Reciprocal Rank Fusion and optional reranking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Sequence

import numpy as np

from src.config import RETRIEVAL
from src.indexing.embedder import Embedder
from src.ingestion.document import LegalChunk
from src.retrieval.case_targeting import canonical_chunk_key, case_match_rank, extract_target_case_name


@dataclass(frozen=True)
class _SearchHit:
    chunk_id: str
    score: float
    metadata: dict


@dataclass
class _FusedCandidate:
    chunk: LegalChunk
    metadata: dict
    rrf_score: float = 0.0
    best_rank: int = 10**9
    rerank_score: float | None = None


class HybridRetriever:
    """Fuse dense and sparse retrieval, then optionally rerank."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        vector_store=None,
        sparse_index=None,
        *,
        bm25_index=None,
        reranker=None,
        dense_k: int | None = None,
        sparse_k: int | None = None,
        rrf_k: int | None = None,
        rerank_k: int | None = None,
    ) -> None:
        resolved_bm25 = bm25_index if bm25_index is not None else sparse_index
        if vector_store is not None and embedder is None:
            raise ValueError("embedder is required when vector_store is provided")

        self.vector_store = vector_store
        self.bm25_index = resolved_bm25
        self.embedder = embedder
        self.reranker = reranker
        self.dense_k = dense_k or RETRIEVAL.dense_k
        self.sparse_k = sparse_k or RETRIEVAL.sparse_k
        self.rrf_k = rrf_k or RETRIEVAL.rrf_k
        self.rerank_k = rerank_k or RETRIEVAL.rerank_k

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        *,
        top_k: int | None = None,
        rrf_k: int | None = None,
    ) -> List[LegalChunk] | list[dict]:
        """Retrieve chunks, or legacy dictionaries when called with top_k."""
        limit = int(top_k if top_k is not None else (k if k is not None else 10))
        if limit <= 0 or not query or not query.strip():
            return []

        use_legacy_output = top_k is not None
        effective_rrf_k = int(rrf_k or self.rrf_k)
        dense_hits = self._dense_search(query, limit if use_legacy_output else self.dense_k)
        sparse_hits = self._sparse_search(query, limit if use_legacy_output else self.sparse_k)
        fused = self._fuse_hits((dense_hits, sparse_hits), rrf_k=effective_rrf_k)
        scoped = self._scope_candidates_to_target_case(query, fused)
        deduped = self._dedupe_candidates(scoped)
        ranked = self._rerank(query, deduped, limit=limit)

        if not ranked:
            return []
        if use_legacy_output:
            return self._legacy_results(ranked, limit)
        return [candidate.chunk for candidate in ranked[:limit]]

    def _dense_search(self, query: str, k: int) -> List[_SearchHit]:
        if self.vector_store is None or self.embedder is None:
            return []

        query_vector = self.embedder.encode([query], normalize=True)[0]
        raw_hits = self.vector_store.search(query_vector, k=k)
        return [self._coerce_hit(item) for item in raw_hits]

    def _sparse_search(self, query: str, k: int) -> List[_SearchHit]:
        if self.bm25_index is None:
            return []

        raw_hits = self.bm25_index.search(query, k=k)
        return [self._coerce_hit(item) for item in raw_hits]

    def _fuse_hits(
        self,
        rank_lists: Sequence[Sequence[_SearchHit]],
        *,
        rrf_k: int,
    ) -> List[_FusedCandidate]:
        candidates: dict[str, _FusedCandidate] = {}

        for hits in rank_lists:
            for rank, hit in enumerate(hits, start=1):
                candidate = candidates.get(hit.chunk_id)
                if candidate is None:
                    candidate = _FusedCandidate(
                        chunk=self._chunk_from_metadata(hit.metadata),
                        metadata=dict(hit.metadata),
                    )
                    candidates[hit.chunk_id] = candidate
                candidate.rrf_score += 1.0 / (rrf_k + rank)
                candidate.best_rank = min(candidate.best_rank, rank)

        return sorted(
            candidates.values(),
            key=lambda item: (-item.rrf_score, item.best_rank, item.chunk.id),
        )

    def _scope_candidates_to_target_case(
        self,
        query: str,
        candidates: Sequence[_FusedCandidate],
    ) -> List[_FusedCandidate]:
        ordered = list(candidates)
        target_case = extract_target_case_name(query)
        if not target_case:
            return ordered

        ranked_matches = [
            (case_match_rank(target_case, candidate.metadata.get("case_name")), index, candidate)
            for index, candidate in enumerate(ordered)
        ]
        matches = [
            candidate
            for match_rank, _, candidate in sorted(
                ranked_matches,
                key=lambda item: (-item[0], item[1]),
            )
            if match_rank > 0
        ]
        return matches

    def _dedupe_candidates(self, candidates: Sequence[_FusedCandidate]) -> List[_FusedCandidate]:
        deduped: list[_FusedCandidate] = []
        seen_keys: set[str] = set()

        for candidate in candidates:
            metadata = candidate.metadata
            key = canonical_chunk_key(
                case_name=metadata.get("case_name"),
                chunk_index=metadata.get("chunk_index"),
                text=metadata.get("text"),
                citation=metadata.get("citation"),
                date_decided=metadata.get("date_decided"),
                court_level=metadata.get("court_level"),
                doc_id=metadata.get("doc_id"),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(candidate)

        return deduped

    def _rerank(
        self,
        query: str,
        candidates: Sequence[_FusedCandidate],
        *,
        limit: int,
    ) -> List[_FusedCandidate]:
        ordered = list(candidates)
        if self.reranker is None or not ordered:
            return ordered

        rerank_count = min(len(ordered), max(limit, self.rerank_k))
        subset = list(ordered[:rerank_count])
        tail = list(ordered[rerank_count:])

        try:
            scores = self._score_candidates(query, subset)
        except Exception:
            return ordered

        if len(scores) != len(subset):
            return ordered

        ranked: list[_FusedCandidate] = []
        for candidate, score in zip(subset, scores):
            candidate.rerank_score = float(score)
            ranked.append(candidate)

        ranked.sort(
            key=lambda item: (
                item.rerank_score if item.rerank_score is not None else float("-inf"),
                item.rrf_score,
                -item.best_rank,
            ),
            reverse=True,
        )
        return ranked + tail

    def _score_candidates(self, query: str, candidates: Sequence[_FusedCandidate]) -> List[float]:
        chunks = [candidate.chunk for candidate in candidates]

        if hasattr(self.reranker, "score"):
            scores = self.reranker.score(query, chunks)
        elif hasattr(self.reranker, "predict"):
            pairs = [(query, chunk.text) for chunk in chunks]
            scores = self.reranker.predict(pairs)
        else:
            raise TypeError("reranker must expose score() or predict()")

        return [float(score) for score in scores]

    @staticmethod
    def _coerce_hit(item) -> _SearchHit:
        if isinstance(item, LegalChunk):
            return _SearchHit(chunk_id=item.id, score=0.0, metadata=item.to_dict())

        if isinstance(item, tuple) and len(item) == 3:
            chunk_id, score, metadata = item
            if not isinstance(metadata, dict):
                raise TypeError("search hit metadata must be a dict")
            payload = HybridRetriever._normalize_metadata(str(chunk_id), metadata)
            return _SearchHit(chunk_id=str(chunk_id), score=float(score), metadata=payload)

        raise TypeError(f"Unsupported search hit type: {type(item)!r}")

    @staticmethod
    def _normalize_metadata(chunk_id: str, metadata: dict) -> dict:
        payload = dict(metadata)
        payload.setdefault("id", chunk_id)
        payload.setdefault("doc_id", chunk_id)
        payload.setdefault("text", "")
        payload.setdefault("chunk_index", 0)
        payload.setdefault("doc_type", "case")
        return payload

    @staticmethod
    def _chunk_from_metadata(metadata: dict) -> LegalChunk:
        raw_date = metadata.get("date_decided")
        parsed_date = None
        if raw_date:
            try:
                parsed_date = date.fromisoformat(str(raw_date))
            except ValueError:
                parsed_date = None

        chunk = LegalChunk(
            id=str(metadata["id"]),
            doc_id=str(metadata["doc_id"]),
            text=str(metadata.get("text", "")),
            chunk_index=int(metadata.get("chunk_index", 0)),
            doc_type=str(metadata.get("doc_type", "case")),
            court_level=metadata.get("court_level"),
            citation=metadata.get("citation"),
            date_decided=parsed_date,
            section_type=metadata.get("section_type"),
            embedding=HybridRetriever._coerce_embedding(metadata.get("embedding")),
        )
        for key in ("case_name", "court", "title", "section", "source_file"):
            value = metadata.get(key)
            if value is not None:
                setattr(chunk, key, value)
        return chunk

    @staticmethod
    def _coerce_embedding(value) -> np.ndarray | None:
        if value is None:
            return None
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def _legacy_results(candidates: Sequence[_FusedCandidate], limit: int) -> list[dict]:
        results: list[dict] = []
        for candidate in candidates[:limit]:
            metadata = candidate.metadata
            results.append(
                {
                    "id": candidate.chunk.id,
                    "doc_id": candidate.chunk.doc_id,
                    "rrf_score": candidate.rrf_score,
                    "rerank_score": candidate.rerank_score,
                    "text": candidate.chunk.text or metadata.get("text", "[No text found]"),
                    "source": metadata.get("source_file", metadata.get("source", "API Corpus")),
                    "citation": metadata.get("citation", "N/A"),
                    "doc_type": candidate.chunk.doc_type,
                    "court_level": candidate.chunk.court_level,
                }
            )
        return results
