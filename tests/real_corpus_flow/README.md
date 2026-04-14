# Real Corpus Flow Tests

These tests exercise the actual `src/` ingestion, chunking, preparation, indexing,
and retrieval code against 10 real CourtListener documents already present in
`data/raw/scotus_cases.jsonl`.

What they cover:

- raw JSONL -> `LegalDocument` loading
- `LegalDocument` -> `LegalChunk` chunking
- raw subset -> processed chunk JSONL preparation
- processed chunk JSONL -> BM25 + Chroma indices
- dense + sparse retrieval over the built indices

The tests intentionally avoid network access. For the dense index path they use a
deterministic local test embedder while still exercising the real `build_indices`,
`ChromaStore`, `BM25Index`, and `HybridRetriever` implementations from `src/`.
