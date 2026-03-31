import tempfile
import numpy as np
import os
from src.indexing.chroma_store import ChromaStore
from src.indexing.bm25_index import BM25Index

def test_chroma_store_retrieval():
    """Test if ChromaDB stores and retrieves the closest vector."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChromaStore(path=tmpdir, collection_name="test_collection")
        
        # Two dummy embeddings. One is small, one is large.
        embeddings = np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]])
        ids = ["doc_small", "doc_large"]
        metadatas = [{"source": "small.pdf"}, {"source": "large.pdf"}]
        
        store.add(ids=ids, embeddings=embeddings, metadata=metadatas)
        
        # Query with a large vector; it should match "doc_large"
        query_vector = np.array([0.8, 0.8, 0.8])
        results = store.search(query_vector, k=1)
        
        # Results format: [(id, distance, metadata)]
        assert len(results) == 1
        assert results[0][0] == "doc_large"
        assert results[0][2]["source"] == "large.pdf"

def test_bm25_exact_match():
    """Test if BM25 correctly scores exact keyword matches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "bm25.pkl")
        index = BM25Index(save_path=save_path)
        
        texts = [
            "The quick brown fox jumps.",
            "This is a legal document regarding a family trust."
        ]
        metadatas = [{"id": 0}, {"id": 1}]
        
        index.build(texts, metadatas)
        
        # Searching for "trust" should return the second document
        results = index.search("trust", k=1)
        
        assert len(results) == 1
        # BM25 returns the string index as the ID by default in our implementation
        assert results[0][0] == "1"