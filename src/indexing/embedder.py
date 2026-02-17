"""
Embedder - sentence-transformers wrapper.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from src.config import MODELS


class Embedder:
    """Simple local embedding wrapper around sentence-transformers."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or MODELS.embedding_model
        self._model = None

    def _load_model(self):
        """Lazy-load to avoid model startup cost until embeddings are needed."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return vectors / norms

    def encode(
        self,
        texts: Sequence[str] | Iterable[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode texts into a 2D float32 array of shape [n, dim]."""
        items = list(texts)
        if not items:
            return np.empty((0, MODELS.embedding_dim), dtype=np.float32)

        model = self._load_model()
        vectors = model.encode(
            items,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if normalize:
            vectors = self._normalize(vectors)

        return vectors
