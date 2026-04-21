"""Ollama local backend."""

from __future__ import annotations

import logging
import time
from typing import Any, List, Optional

import requests

from src.config import LLM
from src.generation.base_llm import BaseLLM
from src.generation.prompts import build_general_legal_prompt, build_rag_legal_prompt


logger = logging.getLogger(__name__)


class OllamaBackendError(RuntimeError):
    """Raised when the Ollama server returns an actionable runtime failure."""


class OllamaBackend(BaseLLM):
    """Minimal Ollama HTTP wrapper used by the API pipeline."""

    def __init__(
        self,
        model: str | None = None,
        *,
        model_name: str | None = None,
        host: str | None = None,
        timeout: int | None = None,
        num_ctx: int | None = None,
        num_batch: int | None = None,
        num_gpu: int | None = None,
    ) -> None:
        self.model = model or model_name or LLM.model
        self.model_name = self.model
        self.host = (host or LLM.host).rstrip("/")
        self.timeout = timeout or LLM.request_timeout_seconds
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
        started = time.perf_counter()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": self._build_options(max_tokens=max_tokens),
        }
        if self.model.lower().startswith("deepseek-r1"):
            # Keep answer text in the standard Ollama "response" field.
            payload["think"] = False
        logger.info(
            "llm.generate_start model=%s host=%s prompt_chars=%s max_tokens=%s",
            self.model,
            self.host,
            len(prompt),
            max_tokens or LLM.max_tokens,
        )
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            message = self._extract_error_message(response)
            logger.warning(
                "llm.generate_error model=%s host=%s elapsed_ms=%.1f error_type=%s message=%r",
                self.model,
                self.host,
                (time.perf_counter() - started) * 1000,
                exc.__class__.__name__,
                message,
            )
            raise OllamaBackendError(
                f"Ollama generate failed for model '{self.model}' at '{self.host}': {message}"
            ) from exc
        except requests.RequestException as exc:
            logger.warning(
                "llm.generate_error model=%s host=%s elapsed_ms=%.1f error_type=%s message=%r",
                self.model,
                self.host,
                (time.perf_counter() - started) * 1000,
                exc.__class__.__name__,
                str(exc),
            )
            raise OllamaBackendError(
                f"Ollama request failed for model '{self.model}' at '{self.host}': {exc}"
            ) from exc
        data = response.json()
        output = str(data.get("response", "")).strip()
        logger.info(
            "llm.generate_complete model=%s host=%s elapsed_ms=%.1f response_chars=%s",
            self.model,
            self.host,
            (time.perf_counter() - started) * 1000,
            len(output),
        )
        return output

    def generate_with_context(
        self,
        query: str,
        context: List[str],
        max_tokens: Optional[int] = None,
        *,
        conversation_history: list[dict[str, Any]] | None = None,
        case_posture: dict[str, Any] | None = None,
        response_depth: str = "concise",
    ) -> str:
        prompt = build_rag_legal_prompt(
            query,
            context,
            conversation_history=conversation_history,
            case_posture=case_posture,
            response_depth=response_depth,
        )
        logger.info(
            "llm.generate_with_context query_chars=%s context_items=%s context_chars=%s history_messages=%s posture=%s response_depth=%s",
            len(query),
            len(context),
            sum(len(item) for item in context),
            len(conversation_history or []),
            "yes" if case_posture else "no",
            response_depth,
        )
        return self.generate(prompt, max_tokens=max_tokens)

    def generate_legal_answer(
        self,
        query: str,
        *,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Convenience wrapper for non-retrieval legal prompting."""
        prompt = build_general_legal_prompt(
            query,
            conversation_history=conversation_history,
        )
        logger.info(
            "llm.generate_direct query_chars=%s history_messages=%s",
            len(query),
            len(conversation_history or []),
        )
        return self.generate(prompt)

    def health_check(self) -> bool:
        """Best-effort Ollama reachability check."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False

    def diagnostics(self) -> dict[str, Any]:
        """Return configured host/model and reachability diagnostics."""
        details: dict[str, Any] = {
            "provider": "ollama",
            "host": self.host,
            "model": self.model,
            "reachable": False,
        }
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            details["error"] = str(exc)
            return details

        details["reachable"] = True
        details["available_models"] = [
            item.get("name")
            for item in payload.get("models", [])
            if item.get("name")
        ]
        details["model_available"] = self.model in details["available_models"]
        return details

    @staticmethod
    def _extract_error_message(response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            error_message = payload.get("error")
            if error_message:
                return str(error_message)

        body = response.text.strip()
        if body:
            return body
        return f"HTTP {response.status_code}"
