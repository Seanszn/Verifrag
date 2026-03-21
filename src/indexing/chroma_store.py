"""
ChromaDB vector index (local).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.config import VECTOR_STORE
from src.indexing.base_store import BaseVectorStore

try:
    import chromadb
except ImportError as exc:  # pragma: no cover - exercised only when dependency is absent
    chromadb = None
    _CHROMA_IMPORT_ERROR = exc
else:
    _CHROMA_IMPORT_ERROR = None


class ChromaStore(BaseVectorStore):
    """Persistent local ChromaDB-backed vector store."""

    def __init__(
        self,
        *,
        path: Path | str | None = None,
        collection_name: str | None = None,
        distance_metric: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        if chromadb is None:  # pragma: no cover - depends on environment
            raise ImportError("chromadb is required to use ChromaStore") from _CHROMA_IMPORT_ERROR

        self.path = Path(path or VECTOR_STORE.chroma_path)
        self.collection_name = collection_name or VECTOR_STORE.chroma_collection
        self.distance_metric = (distance_metric or VECTOR_STORE.chroma_distance).lower()
        self.batch_size = max(1, int(batch_size or VECTOR_STORE.chroma_batch_size))
        self._client = None
        self._collection = None
        self.load()

    def add(self, ids: List[str], embeddings: np.ndarray, metadata: List[dict]) -> None:
        """Add vectors, documents, and metadata to the Chroma collection."""
        if not ids:
            return

        if len(ids) != len(metadata):
            raise ValueError("ids and metadata must have the same length")

        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if len(ids) != len(vectors):
            raise ValueError("ids and embeddings must have the same length")

        collection = self._ensure_collection()
        for start in range(0, len(ids), self.batch_size):
            end = start + self.batch_size
            batch_ids = [str(item) for item in ids[start:end]]
            batch_vectors = vectors[start:end].tolist()
            batch_metadata = metadata[start:end]
            batch_documents: List[str] = []
            batch_metadatas: List[dict] = []

            for item_id, item in zip(batch_ids, batch_metadata):
                if not isinstance(item, dict):
                    raise TypeError("metadata entries must be dictionaries")

                text = item.get("text")
                if text is None:
                    raise ValueError("metadata entries must include text")

                batch_documents.append(str(text))
                batch_metadatas.append(self._sanitize_metadata(item, item_id))

            collection.add(
                ids=batch_ids,
                embeddings=batch_vectors,
                documents=batch_documents,
                metadatas=batch_metadatas,
            )

    def search(self, query_embedding: np.ndarray, k: int) -> List[Tuple[str, float, dict]]:
        """Search for nearest vectors and return (id, score, metadata) tuples."""
        if k <= 0:
            return []

        collection = self._ensure_collection()
        if hasattr(collection, "count") and collection.count() == 0:
            return []

        vector = np.asarray(query_embedding, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        raw = collection.query(
            query_embeddings=vector.tolist(),
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        ids = raw.get("ids") or [[]]
        documents = raw.get("documents") or [[]]
        metadatas = raw.get("metadatas") or [[]]
        distances = raw.get("distances") or [[]]

        results: List[Tuple[str, float, dict]] = []
        for chunk_id, document, metadata, distance in zip(
            ids[0],
            documents[0],
            metadatas[0],
            distances[0],
        ):
            payload = dict(metadata or {})
            payload["id"] = payload.get("id", str(chunk_id))
            payload["text"] = document
            score = self._score_from_distance(float(distance))
            results.append((str(chunk_id), score, payload))

        return results

    def delete(self, ids: List[str]) -> None:
        """Delete vectors by ID."""
        if not ids:
            return
        self._ensure_collection().delete(ids=[str(item) for item in ids])

    def save(self) -> None:
        """Persist the index to disk."""
        # Chroma's persistent client flushes state automatically.
        return None

    def load(self) -> None:
        """Load or create the persistent collection."""
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.path))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric},
        )

    def _ensure_collection(self):
        if self._collection is None:
            self.load()
        return self._collection

    def _score_from_distance(self, distance: float) -> float:
        if self.distance_metric == "cosine":
            return 1.0 - distance
        return -distance

    @staticmethod
    def _sanitize_metadata(metadata: dict, fallback_id: str) -> dict:
        payload = {"id": str(metadata.get("id", fallback_id))}
        for key, value in metadata.items():
            if key in {"text", "embedding"} or value is None:
                continue
            payload[key] = ChromaStore._coerce_metadata_value(value)
        return payload

    @staticmethod
    def _coerce_metadata_value(value):
        if isinstance(value, (str, bool, int, float)):
            return value
        return str(value)
