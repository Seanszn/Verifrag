import chromadb
import numpy as np
from typing import List, Tuple
from src.indexing.base_store import BaseVectorStore

class ChromaStore(BaseVectorStore):
    """Local ChromaDB implementation for vector search."""

    def __init__(self, path: str = "./data/index/chroma", collection_name: str = "legal_verifrag"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add(self, ids: List[str], embeddings: np.ndarray, metadata: List[dict]) -> None:
        # Chroma expects embeddings as a list of lists (floats)
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadata
        )

    def search(self, query_embedding: np.ndarray, k: int) -> List[Tuple[str, float, dict]]:
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )
        
        # Flatten results to match your (id, score, metadata) interface
        search_results = []
        for i in range(len(results['ids'][0])):
            search_results.append((
                results['ids'][0][i],
                results['distances'][0][i] if results['distances'] else 0.0,
                results['metadatas'][0][i]
            ))
        return search_results

    def delete(self, ids: List[str]) -> None:
        self.collection.delete(ids=ids)

    def save(self) -> None:
        # PersistentClient saves automatically in newer Chroma versions
        pass

    def load(self) -> None:
        # Managed by PersistentClient init
        pass