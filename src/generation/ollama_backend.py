"""Ollama local backend implementation."""

from __future__ import annotations

from typing import List, Optional

import requests

from src.config import LLM
from src.generation.base_llm import BaseLLM
from src.generation.prompts import build_general_legal_prompt, build_rag_legal_prompt


class OllamaBackend(BaseLLM):
    """Minimal Ollama HTTP wrapper used by the API pipeline."""

    def __init__(
        self,
        model: str | None = None,
        *,
        model_name: str | None = None,
        host: str | None = None,
        timeout: int = 120,
        num_ctx: int | None = None,
        num_batch: int | None = None,
        num_gpu: int | None = None,
    ) -> None:
        self.model = model or model_name or LLM.model
        self.model_name = self.model
        self.host = (host or LLM.host).rstrip("/")
        self.timeout = timeout
        self.num_ctx = LLM.num_ctx if num_ctx is None else num_ctx
        self.num_batch = LLM.num_batch if num_batch is None else num_batch
        self.num_gpu = LLM.num_gpu if num_gpu is None else num_gpu

    def _build_options(self, max_tokens: Optional[int] = None) -> dict[str, int | float]:
        options: dict[str, int | float] = {
            "temperature": LLM.temperature,
            "num_predict": max_tokens or LLM.max_tokens,
        }
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        if self.num_batch is not None:
            options["num_batch"] = self.num_batch
        if self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu
        return options

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": self._build_options(max_tokens=max_tokens),
        }
        if self.model.lower().startswith("deepseek-r1"):
            payload["think"] = False

        response = requests.post(
            f"{self.host}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return str(data.get("response", "")).strip()

    def generate_legal_answer(self, query: str, max_tokens: Optional[int] = None) -> str:
        prompt = build_general_legal_prompt(query)
        return self.generate(prompt, max_tokens=max_tokens)

    def generate_with_context(
        self,
        query: str,
        context: List[str],
        max_tokens: Optional[int] = None,
    ) -> str:
        prompt = build_rag_legal_prompt(query, context)
        return self.generate(prompt, max_tokens=max_tokens)
