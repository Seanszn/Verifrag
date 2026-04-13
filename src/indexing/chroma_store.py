"""ChromaDB vector index (local)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import VECTOR_STORE
from src.indexing.base_store import BaseVectorStore

try:
    import chromadb
except ImportError:  # pragma: no cover - exercised via runtime dependency checks
    chromadb = None


class ChromaStore(BaseVectorStore):
    """Persistent local vector store backed by ChromaDB."""

    def __init__(
        self,
        path: str | Path = VECTOR_STORE.chroma_path,
        *,
        collection_name: str | None = None,
        distance: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.collection_name = collection_name or VECTOR_STORE.chroma_collection
        self.distance = distance or VECTOR_STORE.chroma_distance
        self.batch_size = int(batch_size or VECTOR_STORE.chroma_batch_size)
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = None
        self.collection = None
        self._client = None
        self._collection = None
        self.load()

    def add(self, ids: list[str], embeddings: np.ndarray, metadata: list[dict]) -> None:
        if len(ids) != len(metadata):
            raise ValueError("ids and metadata must have matching lengths")
        if len(ids) != len(embeddings):
            raise ValueError("ids and embeddings must have matching lengths")
        if not ids:
            return

        collection = self._ensure_collection()
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("embeddings must be a 2D array")

        for start in range(0, len(ids), self.batch_size):
            end = start + self.batch_size
            batch_ids = [str(item) for item in ids[start:end]]
            batch_vectors = vectors[start:end]
            batch_metadata = [self._sanitize_metadata(item) for item in metadata[start:end]]
            batch_documents = [item.get("text", "") for item in batch_metadata]
            collection.add(
                ids=batch_ids,
                embeddings=batch_vectors.tolist(),
                documents=batch_documents,
                metadatas=batch_metadata,
            )

    def search(self, query_embedding: np.ndarray, k: int) -> list[tuple[str, float, dict]]:
        if k <= 0:
            return []

        collection = self._ensure_collection()
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim != 1:
            raise ValueError("query_embedding must be a 1D array")

        response = collection.query(
            query_embeddings=[query.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: list[tuple[str, float, dict]] = []
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            payload = dict(metadata or {})
            if "text" not in payload:
                payload["text"] = document
            similarity = 1.0 - float(distance) if distance is not None else 0.0
            results.append((str(chunk_id), similarity, payload))
        return results

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        collection = self._ensure_collection()
        collection.delete(ids=[str(item) for item in ids])

    def save(self) -> None:
        # Chroma persistence is handled by the client.
        self._ensure_collection()

    def load(self) -> None:
        self._client = self._make_client()
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance},
        )
        self.client = self._client
        self.collection = self._collection

    def _ensure_collection(self):
        if self._collection is None:
            self.load()
        return self._collection

    def _make_client(self):
        if chromadb is None:
            raise RuntimeError(
                "chromadb is not installed. Install the project requirements to enable vector search."
            )
        return chromadb.PersistentClient(path=str(self.path))

    @staticmethod
    def _sanitize_metadata(metadata: dict) -> dict:
        payload: dict = {}
        for key, value in dict(metadata).items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                payload[key] = value
            else:
                payload[key] = str(value)
        return payload
