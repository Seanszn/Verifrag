"""
Abstract LLM interface.
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseLLM(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def generate_with_context(self, query: str, context: List[str], max_tokens: Optional[int] = None) -> str:
        """Generate a response with retrieved context."""
        pass

    def generate_legal_answer(self, query: str, max_tokens: Optional[int] = None) -> str:
        """Compatibility wrapper for direct legal analysis calls."""
        return self.generate(query, max_tokens=max_tokens)
