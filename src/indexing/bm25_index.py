import pickle
import os
from rank_bm25 import BM25Okapi
from typing import List, Tuple
import numpy as np

class BM25Index:
    """Keyword-based sparse index using BM25."""

    def __init__(self, save_path: str = "./data/index/bm25.pkl"):
        self.save_path = save_path
        self.bm25 = None
        self.corpus = []  # List of raw text strings
        self.metadatas = []

    def build(self, texts: List[str], metadatas: List[dict]):
        self.corpus = texts
        self.metadatas = metadatas
        # Tokenize for BM25
        tokenized_corpus = [doc.lower().split() for doc in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.save()

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, dict]]:
        if not self.bm25:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_n:
            results.append((
                str(idx), 
                float(scores[idx]), 
                self.metadatas[idx]
            ))
        return results

    def save(self):
        with open(self.save_path, 'wb') as f:
            pickle.dump((self.corpus, self.metadatas, self.bm25), f)

    def load(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, 'rb') as f:
                self.corpus, self.metadatas, self.bm25 = pickle.load(f)