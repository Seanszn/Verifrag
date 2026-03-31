import tempfile
import os
import numpy as np
from src.indexing.embedder import Embedder
from src.indexing.chroma_store import ChromaStore
from src.indexing.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever

def test_hybrid_retrieval_rrf():
    """Test that HybridRetriever merges Chroma and BM25 results using RRF."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Initialize our stores
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        vector_store = ChromaStore(path=tmpdir, collection_name="test_hybrid")
        
        bm25_path = os.path.join(tmpdir, "bm25.pkl")
        sparse_index = BM25Index(save_path=bm25_path)
        
        # 2. Create mock data
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "The Supreme Court ruled on the family trust taxation.",
            "Apples are subject to a tariff."
        ]
        ids = ["doc_1", "doc_2", "doc_3"]
        metadatas = [
            {"text": texts[0], "source_file": "fox.pdf"},
            {"text": texts[1], "source_file": "trust.pdf"},
            {"text": texts[2], "source_file": "apples.pdf"}
        ]
        
        # 3. Populate Indexes
        embeddings = embedder.encode(texts)
        vector_store.add(ids=ids, embeddings=embeddings, metadata=metadatas)
        sparse_index.build(texts=texts, metadatas=metadatas)
        
        # 4. Initialize Retriever
        retriever = HybridRetriever(
            embedder=embedder,
            vector_store=vector_store,
            sparse_index=sparse_index
        )
        
        # 5. Perform a query heavily targeting document 2
        query = "Supreme Court family trust"
        results = retriever.retrieve(query=query, top_k=2)
        
        # 6. Assertions
        assert len(results) > 0, "Retriever should return results."
        
        # The top result should be doc_2 because it has exact keyword matches AND semantic meaning
        top_result = results[0]
        assert top_result["id"] == "doc_2"
        assert "family trust" in top_result["text"]
        
        # Verify RRF score exists and is formatted correctly
        assert "rrf_score" in top_result
        assert top_result["rrf_score"] > 0.0