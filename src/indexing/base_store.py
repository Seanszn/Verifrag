"""
Abstract vector store interface.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add(self, ids: List[str], embeddings: np.ndarray, metadata: List[dict]) -> None:
        """Add vectors to the store."""
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int) -> List[Tuple[str, float, dict]]:
        """Search for similar vectors. Returns list of (id, score, metadata)."""
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete vectors by ID."""
        pass

    @abstractmethod
    def save(self) -> None:
        """Persist the index to disk."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load the index from disk."""
        pass
