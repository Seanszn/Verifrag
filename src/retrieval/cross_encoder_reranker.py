"""Cross-encoder reranker for second-stage retrieval scoring."""

from __future__ import annotations

from typing import Sequence

from src.config import MODELS


class CrossEncoderReranker:
    """Lazy cross-encoder wrapper used to rerank retrieved chunks."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name or MODELS.rerank_model
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def score(self, query: str, chunks: Sequence[object]) -> list[float]:
        if not chunks:
            return []

        model = self._load_model()
        pairs = [(query, self._chunk_text(chunk)) for chunk in chunks]
        scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]

    @staticmethod
    def _chunk_text(chunk: object) -> str:
        text = getattr(chunk, "text", None)
        if text is not None:
            return str(text)
        if isinstance(chunk, dict):
            return str(chunk.get("text", ""))
        return str(chunk)
