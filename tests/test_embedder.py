"""Tests for local embedding wrapper."""

import numpy as np

from src.config import MODELS
from src.indexing.embedder import Embedder


class _FakeModel:
    def encode(self, items, batch_size=32, convert_to_numpy=True, show_progress_bar=False):
        _ = batch_size, convert_to_numpy, show_progress_bar
        return np.array([[3.0, 4.0, 0.0] for _ in items], dtype=np.float32)


def test_encode_returns_normalized_float32_vectors():
    embedder = Embedder()
    embedder._model = _FakeModel()

    vectors = embedder.encode(["alpha", "beta"], batch_size=2, normalize=True)

    assert vectors.shape == (2, 3)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0], rtol=1e-6)


def test_encode_empty_input_returns_empty_matrix_with_config_dim():
    embedder = Embedder()
    vectors = embedder.encode([], normalize=True)

    assert vectors.shape == (0, MODELS.embedding_dim)
    assert vectors.dtype == np.float32
