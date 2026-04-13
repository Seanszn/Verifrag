"""Ollama local backend."""

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
    ) -> None:
        self.model = model or model_name or LLM.model
        self.model_name = self.model
        self.host = (host or LLM.host).rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": LLM.temperature,
                "num_predict": max_tokens or LLM.max_tokens,
            },
        }
        if self.model.lower().startswith("deepseek-r1"):
            # Keep answer text in the standard Ollama "response" field.
            payload["think"] = False
        response = requests.post(
            f"{self.host}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return str(data.get("response", "")).strip()

    def generate_with_context(
        self,
        query: str,
        context: List[str],
        max_tokens: Optional[int] = None,
    ) -> str:
        prompt = build_rag_legal_prompt(query, context)
        return self.generate(prompt, max_tokens=max_tokens)

    def generate_legal_answer(self, query: str) -> str:
        """Convenience wrapper for non-retrieval legal prompting."""
        return self.generate(build_general_legal_prompt(query))

    def health_check(self) -> bool:
        """Best-effort Ollama reachability check."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False
