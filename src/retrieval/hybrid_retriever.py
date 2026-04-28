"""Hybrid retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import logging
from typing import List, Sequence

import numpy as np

from src.config import RETRIEVAL
from src.indexing.embedder import Embedder
from src.ingestion.document import LegalChunk
from src.retrieval.query_expansion import QueryExpansionConfig, QueryExpansionPlan, build_query_expansion_plan

logger = logging.getLogger(__name__)

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
        query_expansion_mode: str | None = None,
        query_expansion_config: QueryExpansionConfig | None = None,
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
        self.query_expansion_config = query_expansion_config or QueryExpansionConfig(
            mode=query_expansion_mode or RETRIEVAL.query_expansion_mode,
            max_variants=RETRIEVAL.query_expansion_max_variants,
            max_terms=RETRIEVAL.query_expansion_max_terms,
        )
        self.last_dense_error: str | None = None
        self.last_sparse_error: str | None = None
        self._last_query_expansion_plan = QueryExpansionPlan(
            "",
            self.query_expansion_config.mode,
            ("",),
            status="not_applied:not_run",
        )
        self._last_retrieval_variant_log: list[dict] = []

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

        self.last_dense_error = None
        self.last_sparse_error = None
        self._last_retrieval_variant_log = []
        use_legacy_output = top_k is not None
        effective_rrf_k = int(rrf_k or self.rrf_k)
        dense_k = limit if use_legacy_output else self.dense_k
        sparse_k = limit if use_legacy_output else self.sparse_k
        raw_dense_hits = self._safe_dense_search(query, dense_k)
        raw_sparse_hits = self._safe_sparse_search(query, sparse_k)

        if self.last_dense_error and self.last_sparse_error:
            raise RuntimeError(
                "dense and sparse retrieval failed: "
                f"dense={self.last_dense_error}; sparse={self.last_sparse_error}"
            )

        rank_lists: list[Sequence[_SearchHit]] = [raw_dense_hits, raw_sparse_hits]
        self._log_variant_hits("raw", query, raw_dense_hits, raw_sparse_hits)
        plan = build_query_expansion_plan(
            query,
            config=self.query_expansion_config,
            corpus_metadata=[hit.metadata for hit in (*raw_dense_hits, *raw_sparse_hits)],
        )
        self._last_query_expansion_plan = plan
        for variant in plan.variants[1:]:
            dense_hits = self._safe_dense_search(variant, dense_k)
            sparse_hits = self._safe_sparse_search(variant, sparse_k)
            rank_lists.extend((dense_hits, sparse_hits))
            self._log_variant_hits("expanded", variant, dense_hits, sparse_hits)

        if self.last_dense_error and self.last_sparse_error and not any(rank_lists):
            raise RuntimeError(
                "dense and sparse retrieval failed: "
                f"dense={self.last_dense_error}; sparse={self.last_sparse_error}"
            )

        fused = self._fuse_hits(rank_lists, rrf_k=effective_rrf_k)

        if not fused:
            return []
        if use_legacy_output:
            return self._legacy_results(fused, limit)

        chunks = [candidate.chunk for candidate in fused]
        reranked = self._rerank(query, chunks)
        return reranked[:limit]

    def last_query_expansion_meta(self) -> dict:
        return {
            **self._last_query_expansion_plan.to_dict(),
            "retrieved_chunk_ids_per_variant": self._last_retrieval_variant_log,
        }

    def last_backend_status(self) -> str:
        if self.last_dense_error and self.last_sparse_error:
            return f"error:dense:{self.last_dense_error};sparse:{self.last_sparse_error}"
        if self.last_dense_error:
            return f"warning:dense:{self.last_dense_error}"
        if self.last_sparse_error:
            return f"warning:sparse:{self.last_sparse_error}"
        return "ok"

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

    def _safe_dense_search(self, query: str, k: int) -> List[_SearchHit]:
        try:
            return self._dense_search(query, k)
        except Exception as exc:
            self.last_dense_error = exc.__class__.__name__
            logger.warning(
                "retrieval.dense_fallback error=%s query=%r",
                self.last_dense_error,
                " ".join(query.split())[:80],
            )
            return []

    def _safe_sparse_search(self, query: str, k: int) -> List[_SearchHit]:
        try:
            return self._sparse_search(query, k)
        except Exception as exc:
            self.last_sparse_error = exc.__class__.__name__
            logger.warning(
                "retrieval.sparse_error error=%s query=%r",
                self.last_sparse_error,
                " ".join(query.split())[:80],
            )
            return []

    def _log_variant_hits(
        self,
        source: str,
        query: str,
        dense_hits: Sequence[_SearchHit],
        sparse_hits: Sequence[_SearchHit],
    ) -> None:
        self._last_retrieval_variant_log.append(
            {
                "source": source,
                "query": query,
                "dense_chunk_ids": [hit.chunk_id for hit in dense_hits],
                "sparse_chunk_ids": [hit.chunk_id for hit in sparse_hits],
                "document_ids": sorted(
                    {
                        str(hit.metadata.get("doc_id") or hit.chunk_id)
                        for hit in (*dense_hits, *sparse_hits)
                    }
                ),
            }
        )

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
                else:
                    candidate.metadata = self._merge_metadata(candidate.metadata, hit.metadata)
                    candidate.chunk = self._chunk_from_metadata(candidate.metadata)
                candidate.rrf_score += 1.0 / (rrf_k + rank)
                candidate.best_rank = min(candidate.best_rank, rank)

        return sorted(
            candidates.values(),
            key=lambda item: (-item.rrf_score, item.best_rank, item.chunk.id),
        )

    def _rerank(self, query: str, chunks: Sequence[LegalChunk]) -> List[LegalChunk]:
        if self.reranker is None or not chunks:
            return list(chunks)

        if hasattr(self.reranker, "rerank"):
            reranked = self.reranker.rerank(query, list(chunks))
            if not reranked:
                return []
            first = reranked[0]
            if isinstance(first, LegalChunk):
                return list(reranked)
            if isinstance(first, tuple) and len(first) == 2:
                return [chunk for chunk, _ in sorted(reranked, key=lambda item: item[1], reverse=True)]

        if hasattr(self.reranker, "score"):
            scores = self.reranker.score(query, list(chunks))
        elif hasattr(self.reranker, "predict"):
            pairs = [(query, chunk.text) for chunk in chunks]
            scores = self.reranker.predict(pairs)
        else:
            raise TypeError("reranker must expose rerank(), score(), or predict()")

        if len(scores) != len(chunks):
            raise ValueError("reranker returned a score count that does not match the chunk count")

        ranked_pairs = sorted(
            zip(chunks, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [chunk for chunk, _ in ranked_pairs]

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

        return LegalChunk(
            id=str(metadata["id"]),
            doc_id=str(metadata["doc_id"]),
            text=str(metadata.get("text", "")),
            chunk_index=int(metadata.get("chunk_index", 0)),
            doc_type=str(metadata.get("doc_type", "case")),
            case_name=metadata.get("case_name"),
            court=metadata.get("court"),
            court_level=metadata.get("court_level"),
            citation=metadata.get("citation"),
            date_decided=parsed_date,
            title=metadata.get("title"),
            section=metadata.get("section"),
            source_file=metadata.get("source_file"),
            embedding=HybridRetriever._coerce_embedding(metadata.get("embedding")),
        )

    @staticmethod
    def _coerce_embedding(value) -> np.ndarray | None:
        if value is None:
            return None
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def _merge_metadata(existing: dict, incoming: dict) -> dict:
        payload = dict(existing)
        for key, value in incoming.items():
            if key not in payload or HybridRetriever._is_blank_metadata_value(payload.get(key)):
                if not HybridRetriever._is_blank_metadata_value(value):
                    payload[key] = value
        return payload

    @staticmethod
    def _is_blank_metadata_value(value) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        return False

    @staticmethod
    def _legacy_results(candidates: Sequence[_FusedCandidate], limit: int) -> list[dict]:
        results: list[dict] = []
        for candidate in candidates[:limit]:
            metadata = candidate.metadata
            results.append(
                {
                    "id": candidate.chunk.id,
                    "rrf_score": candidate.rrf_score,
                    "text": candidate.chunk.text or metadata.get("text", "[No text found]"),
                    "source": metadata.get("source_file", metadata.get("source", "API Corpus")),
                    "citation": metadata.get("citation", "N/A"),
                }
            )
        return results
