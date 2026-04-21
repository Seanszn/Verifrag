"""
Abstract LLM interface.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BaseLLM(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
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
        """Generate a response with retrieved context."""
        pass
