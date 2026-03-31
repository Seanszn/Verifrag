"""
Hybrid Retriever.
Combines ChromaDB vector search and BM25 sparse search using Reciprocal Rank Fusion.
"""

from typing import List, Dict, Any
from src.indexing.embedder import Embedder
from src.indexing.chroma_store import ChromaStore
from src.indexing.bm25_index import BM25Index

class HybridRetriever:
    def __init__(self, embedder: Embedder, vector_store: ChromaStore, sparse_index: BM25Index):
        self.embedder = embedder
        self.vector_store = vector_store
        self.sparse_index = sparse_index

    def retrieve(self, query: str, top_k: int = 5, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """
        Search both indexes and merge results.
        rrf_k is a constant used in the Reciprocal Rank Fusion formula (60 is standard).
        """
        # 1. Embed the query and get vector results
        query_embedding = self.embedder.encode([query])[0]
        vector_results = self.vector_store.search(query_embedding, k=top_k)
        
        # 2. Get keyword results
        sparse_results = self.sparse_index.search(query, k=top_k)

        # 3. Apply Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        metadata_map = {}

        # Score Vector Rankings
        for rank, (doc_id, score, metadata) in enumerate(vector_results):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
                metadata_map[doc_id] = metadata
            rrf_scores[doc_id] += 1.0 / (rrf_k + rank + 1)

        # Score Keyword Rankings
        for rank, (doc_id, score, metadata) in enumerate(sparse_results):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
                metadata_map[doc_id] = metadata
            rrf_scores[doc_id] += 1.0 / (rrf_k + rank + 1)

        # 4. Sort by highest RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

        # 5. Format and return top_k results for the LLM context
        final_results = []
        for doc_id, score in sorted_docs[:top_k]:
            final_results.append({
                "id": doc_id,
                "rrf_score": score,
                "text": metadata_map[doc_id].get("text", "[No text found]"),
                "source": metadata_map[doc_id].get("source_file", "API Corpus"),
                "citation": metadata_map[doc_id].get("citation", "N/A")
            })

        return final_results