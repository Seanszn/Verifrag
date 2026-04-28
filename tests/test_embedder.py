"""Tests for local embedding wrapper."""

import sys
import types

import numpy as np
import pytest

from src import config as config_module
from src.indexing.embedder import Embedder


pytestmark = pytest.mark.smoke


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

    assert vectors.shape == (0, config_module.MODELS.embedding_dim)
    assert vectors.dtype == np.float32


def test_load_model_forwards_local_files_only(monkeypatch):
    captured = {}

    class _FakeSentenceTransformer:
        def __init__(self, model_name, *, local_files_only=False, **kwargs):
            captured["model_name"] = model_name
            captured["local_files_only"] = local_files_only
            captured["kwargs"] = kwargs

    fake_module = types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(config_module.MODELS, "huggingface_local_files_only", True)

    embedder = Embedder()
    embedder._load_model()

    assert captured["model_name"] == config_module.MODELS.embedding_model
    assert captured["local_files_only"] is True
