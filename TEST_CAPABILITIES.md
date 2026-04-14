# Test-Proven Capabilities

This report describes what the currently passing tests prove about the system.
It is based on the backend/pipeline-oriented green run after wiring retrieval
into generation.

## Test Run Used

Command:

```powershell
.\venv\Scripts\python.exe -B -m pytest -p no:cacheprovider --ignore=tests\test_user_file_ingestion.py --ignore=tests\test_document_processing_demo.py --ignore=tests\test_auth_ui.py -q
```

Result:

```text
46 passed in 14.34s
```

A focused pipeline/API/retrieval run also passed:

```text
10 passed in 5.99s
```

The focused run is covered by the 46-test capability list below.

## Important Scope Limits

These tests prove local code behavior under the tested conditions. They do not
prove every production dependency or every UI flow.

- They do not prove live Ollama response quality or that a local Ollama server is running.
- They do not prove live transformer model download or live NLI model inference; NLI scoring is stubbed in the unit tests.
- They do not prove CourtListener network access; the CourtListener API client test uses a fake `_get` method.
- They do not prove the Streamlit authentication UI currently passes; `tests/test_auth_ui.py` was excluded after unrelated failures.
- They do not prove `tests/test_user_file_ingestion.py`; it imports `src.ingestion.user_file_ingestion`, which is missing in the current tree.
- They do not prove `tests/test_document_processing_demo.py`; it requires missing fixture `demo/fixtures/document_processing_demo.json`.
- They do not prove uploaded documents are automatically added to the active search index after upload; the upload API persists raw and processed upload files, but index rebuild/reload is a separate concern.

## Capability Summary

- The API can register a user, submit a query, persist user and assistant messages, and return conversation history through FastAPI.
- The query pipeline can retrieve evidence before generation and call the RAG generation path with retrieved context when chunks are available.
- The query pipeline can fall back to direct generation when no retriever or no evidence is available.
- The query pipeline can reuse the retrieved chunks for downstream claim verification.
- The default retriever loader can discover active BM25 and Chroma artifacts from index summary metadata.
- Raw legal corpus JSONL records can be loaded, chunked, prepared into processed chunk JSONL files, indexed, and searched.
- BM25 search can rank sparse lexical matches.
- Hybrid retrieval can fuse dense and sparse hits using reciprocal-rank fusion and can accept reranker ordering.
- Chroma vector store integration can add, search, delete, and persist vector metadata in tested configurations.
- The embedder wrapper normalizes non-empty vectors and returns correctly shaped empty arrays without loading a model for empty input.
- Claim decomposition can split certain legal sentences into atomic claims, handle attribution, accept corpus-builder records, and read JSONL document input.
- NLI aggregation can select best evidence using authority weights, batch claim/chunk pairs, flag contradiction penalties, and return empty results for empty evidence.
- Citation verification can extract common case/statute citations, verify citation authority before semantic support scoring, and reject wrong citations even when the proposition is semantically supported elsewhere.
- Upload API authentication is enforced, and authenticated text uploads are parsed, chunked, persisted, and linked to the uploading user workspace.

## Exhaustive Passing Test Evidence

### Real Corpus Flow

- `tests/real_corpus_flow/test_real_10_doc_flow.py::test_step_01_load_first_10_real_documents_and_record_inputs`
  - Proves the system can load a 10-document subset from real CourtListener-style SCOTUS JSONL input.
  - Proves loaded documents include expected legal document shape: IDs start with `cl_`, document type is `case`, and report output can be written.

- `tests/real_corpus_flow/test_real_10_doc_flow.py::test_step_02_chunk_first_real_document_and_record_outputs`
  - Proves a real loaded legal document can be chunked into multiple chunks.
  - Proves chunk IDs follow the `{doc_id}:{chunk_index}` contract and all chunks preserve the source document ID.

- `tests/real_corpus_flow/test_real_10_doc_flow.py::test_step_03_prepare_real_10_doc_subset_and_record_processed_outputs`
  - Proves the corpus preparation layer can turn 10 raw real-case records into processed chunk JSONL rows.
  - Proves the processed output preserves 10 distinct document IDs and writes a preparation summary.

- `tests/real_corpus_flow/test_real_10_doc_flow.py::test_step_04_build_indices_and_retrieve_over_real_10_doc_subset`
  - Proves indices can be built from the real 10-document processed subset.
  - Proves dense search, BM25 search, and fused hybrid retrieval all return hits for the subset.
  - Proves the fused result can retrieve a chunk from the expected source document for a case-name query.

### API and Conversation Flow

- `tests/test_api_app.py::test_register_query_and_history`
  - Proves FastAPI registration returns HTTP 201 and a usable bearer token.
  - Proves authenticated `/api/query` returns HTTP 200 with an assistant message and pipeline metadata containing claims.
  - Proves user and assistant messages are persisted and can be loaded from `/api/conversations/{conversation_id}/messages`.
  - Proves the backend API can run a lightweight query pipeline without live LLM or live retrieval dependencies.

### Index Building

- `tests/test_build_index.py::test_build_indices_creates_persisted_bm25_chroma_and_summary`
  - Proves `build_indices()` loads processed chunk JSONL rows and creates a searchable BM25 artifact.
  - Proves dense vector search returns the expected top chunk with round-tripped metadata under the fake Chroma client.
  - Proves the index summary records chunk count, embedding shape, collection name, and processed input files.

- `tests/test_build_index.py::test_build_index_cli_writes_default_named_artifacts`
  - Proves the `scripts/build_index.py` CLI can build indices from a processed directory.
  - Proves default artifact names include `bm25.pkl` and `index_summary.json`.
  - Proves the CLI reports the number of indexed chunks and processed files.

### Chunking

- `tests/test_chunker.py::test_chunk_document_preserves_overlap_and_metadata`
  - Proves chunking uses the configured word overlap.
  - Proves generated chunk IDs increment from the source document ID.
  - Proves chunk metadata such as `doc_id`, `citation`, and `court_level` is preserved.

- `tests/test_chunker.py::test_chunk_document_handles_headings_without_changing_chunk_contract`
  - Proves heading-like source text still produces chunks.
  - Proves the chunk ID and document ID contract remains stable for uploaded-style documents.

### Citation Verification

- `tests/test_citation_verifier.py::test_chunk_text_splits_into_fixed_size_word_groups`
  - Proves citation-verifier text chunking splits source text into fixed-size word groups.

- `tests/test_citation_verifier.py::test_classify_score_respects_claim_metadata_thresholds`
  - Proves support classification thresholds vary by claim metadata.
  - Proves a lower score can be `SUPPORTED` for an alleged fact while a higher score can be only `PARTIAL` for a found holding.

- `tests/test_citation_verifier.py::test_verify_claim_against_text_returns_best_matching_chunk`
  - Proves claim verification against raw source text selects the best matching source chunk.
  - Proves it returns a `SUPPORTED` verdict, evidence text, and confidence score for a matching claim.

- `tests/test_citation_verifier.py::test_verify_claim_against_text_handles_empty_source_text`
  - Proves empty source text is handled as an error result instead of crashing.
  - Proves the returned error reason is `No source text`.

- `tests/test_citation_verifier.py::test_verify_claims_accepts_decomposed_claims_batch`
  - Proves decomposed claims can be passed into batch citation/support verification.
  - Proves the batch returns one result per claim.
  - Proves the tested decomposed claims are classified as `SUPPORTED` against the fixture text.

- `tests/test_citation_verifier.py::test_extract_citations_finds_case_and_statutory_references`
  - Proves citation extraction detects case citations such as `Miranda v. Arizona`.
  - Proves citation extraction detects statutory references.
  - Proves extracted citation records include citation type and case-name metadata.

- `tests/test_citation_verifier.py::test_verify_citation_claim_matches_authority_before_scoring_support`
  - Proves citation verification checks that the cited authority exists before support scoring.
  - Proves a correct Miranda citation can be matched to the expected document, chunk, and citation string.
  - Proves a correct citation plus supported proposition can produce a `VERIFIED` verdict with high confidence.

- `tests/test_citation_verifier.py::test_verify_citation_claim_rejects_semantically_supported_text_with_wrong_citation`
  - Proves semantically supported text is rejected when the cited authority is wrong or absent.
  - Proves wrong-citation cases return `CITATION_ERROR` with no matched document or chunk.

### Claim Decomposition

- `tests/test_claim_decomposer.py::test_golden_case_conjoined_verbs_become_atomic_claims`
  - Proves conjoined verbs can be split into separate atomic claims.
  - Example proven behavior: `entered the residence` and `arrested Doe` become separate claim texts.

- `tests/test_claim_decomposer.py::test_golden_case_attribution_emits_attribution_and_embedded_claim`
  - Proves attribution statements produce an attribution claim.
  - Proves the embedded factual/legal proposition is also emitted separately.

- `tests/test_claim_decomposer.py::test_split_clauses_conjoined_verbs`
  - Proves clause splitting handles simple conjoined-verb structures.

- `tests/test_claim_decomposer.py::test_split_clauses_conjoined_verbs_with_multiword_subject`
  - Proves clause splitting preserves multiword subjects while separating conjoined verbs.

- `tests/test_claim_decomposer.py::test_corpus_builder_record_format_is_accepted`
  - Proves the decomposer accepts corpus-builder style records.
  - Proves generated claim spans preserve the source document ID.

- `tests/test_claim_decomposer.py::test_load_document_reads_jsonl_first_record`
  - Proves the decomposer document loader can read the first record from JSONL input.

### CourtListener Corpus Builder API Client

- `tests/test_corpus_builder_api.py::test_fetch_clusters_omits_deprecated_ordering_param`
  - Proves `fetch_clusters()` builds a CourtListener clusters endpoint request.
  - Proves it includes court ID, modified-since timestamp, and page size.
  - Proves it omits the deprecated `ordering` parameter.

### Embedding Wrapper

- `tests/test_embedder.py::test_encode_returns_normalized_float32_vectors`
  - Proves the embedder wrapper returns `float32` vectors.
  - Proves normalization produces unit-length vectors when a model returns non-empty vectors.

- `tests/test_embedder.py::test_encode_empty_input_returns_empty_matrix_with_config_dim`
  - Proves empty input returns an empty matrix with the configured embedding dimension.
  - Proves empty input does not require loading the embedding model.

### Index Discovery

- `tests/test_index_discovery.py::test_discover_index_artifacts_reads_suffix_summary`
  - Proves index discovery can read a suffixed summary file such as `nli_100_index_summary.json`.
  - Proves relative BM25 and Chroma paths from the summary are resolved against the summary directory.
  - Proves the Chroma collection name and summary source are preserved.

- `tests/test_index_discovery.py::test_discover_index_artifacts_prefers_canonical_summary`
  - Proves `index_summary.json` takes precedence over suffixed summary files.
  - Proves the canonical summary controls BM25 path, Chroma path, and collection name when present.

### NLI Verification

- `tests/test_nli_verifier.py::test_verify_claim_uses_authority_weighted_best_chunk`
  - Proves NLI aggregation uses authority weighting when choosing the best evidence chunk.
  - Proves a SCOTUS chunk can outrank a district chunk despite a lower raw entailment score.

- `tests/test_nli_verifier.py::test_verify_claims_batch_builds_all_claim_chunk_pairs`
  - Proves batch verification builds every claim/chunk pair for scoring.
  - Proves two claims across two chunks generate four scored pairs.
  - Proves each claim can select its own best supporting chunk.

- `tests/test_nli_verifier.py::test_contradiction_penalty_flags_contradicted_claim`
  - Proves high contradiction probability flags a claim as contradicted.
  - Proves contradiction penalty contributes to a low final score.

- `tests/test_nli_verifier.py::test_empty_chunk_list_returns_empty_result`
  - Proves verification against no evidence returns a safe empty result.
  - Proves empty results have score `0.0`, no best chunk, and best chunk index `-1`.

### Query Pipeline

- `tests/test_pipeline.py::test_pipeline_runs_real_verifier_logic_when_retrieval_is_available`
  - Proves the query pipeline retrieves evidence before generation when a retriever is configured.
  - Proves generation uses `generate_with_context()` rather than direct generation when retrieved chunks exist.
  - Proves the prompt context passed to generation includes citation and evidence text.
  - Proves the same retrieved chunks are reused for downstream NLI verification.
  - Proves pipeline metadata reports `generation_mode: rag`, `retrieval_used: true`, retrieval chunk count, and `verification_backend_status: ok`.
  - Proves verified claim metadata includes best evidence citation and a positive support score.

- `tests/test_pipeline.py::test_pipeline_reports_skipped_verification_without_indices`
  - Proves the pipeline can run without default indices.
  - Proves no-retriever mode falls back to direct generation.
  - Proves metadata reports `generation_mode: direct`, `retrieval_used: false`, zero retrieval chunks, and `skipped:no_retriever`.
  - Proves claim decomposition still runs on the direct generated answer.

- `tests/test_pipeline.py::test_load_default_retriever_uses_discovered_artifacts`
  - Proves the default retriever loader uses discovered index artifacts rather than hard-coded default paths.
  - Proves BM25 path, Chroma path, and Chroma collection name from discovered artifacts are passed into retriever construction.

### Corpus Preparation

- `tests/test_prepare_corpus.py::test_prepare_corpus_writes_processed_chunks_and_summary`
  - Proves raw corpus JSONL records can be converted into processed chunk JSONL.
  - Proves processed rows preserve document ID, citation, and chunk index.
  - Proves legacy `section_type` metadata is not emitted in processed chunk rows.
  - Proves preparation summary records raw input file, output file, and document count.

- `tests/test_prepare_corpus.py::test_prepare_corpus_cli_uses_expected_output_names`
  - Proves the `scripts/prepare_corpus.py` CLI exits successfully for valid raw input.
  - Proves `_cases.jsonl` input names are converted to `_chunks.jsonl` output names.
  - Proves the CLI writes processed chunk files and `prep_summary.json`.

### Retrieval

- `tests/test_retrieval.py::test_bm25_index_returns_matching_chunks_by_score`
  - Proves the BM25 index returns matching chunks ordered by score.
  - Proves lexical matches produce positive scores.

- `tests/test_retrieval.py::test_hybrid_retriever_fuses_dense_and_sparse_hits_with_rrf`
  - Proves hybrid retrieval calls both dense vector search and sparse BM25 search.
  - Proves reciprocal-rank fusion merges dense and sparse hits into a combined order.
  - Proves the query is embedded before dense search.

- `tests/test_retrieval.py::test_hybrid_retriever_reranker_can_override_fused_order`
  - Proves an optional reranker can override the fused retrieval order.
  - Proves only the requested top-k reranked chunks are returned.

- `tests/test_retrieval.py::test_hybrid_retriever_returns_empty_for_blank_query`
  - Proves blank queries return an empty retrieval result.
  - Proves blank queries do not call dense or sparse search backends.

### Upload API

- `tests/test_uploads_api.py::test_upload_requires_authentication`
  - Proves `/api/uploads` rejects unauthenticated requests with HTTP 401.

- `tests/test_uploads_api.py::test_upload_ingests_text_file_to_user_workspace`
  - Proves authenticated text uploads return HTTP 201.
  - Proves upload responses include conversation ID, file count, document count, chunk count, sanitized filename, and privilege flag.
  - Proves uploaded text files are stored in the user workspace.
  - Proves upload ingestion writes raw user-upload JSONL and processed user-upload chunk JSONL.
  - Proves processed upload chunks reference the raw uploaded document ID.

### Vector Store

- `tests/test_vector_stores.py::test_chroma_store_search_returns_ranked_results_with_round_tripped_metadata`
  - Proves Chroma store search returns ranked results by vector similarity under the fake persistent client.
  - Proves stored metadata such as text, document ID, and chunk index round-trips through search.

- `tests/test_vector_stores.py::test_chroma_store_delete_removes_vectors`
  - Proves deleting vector IDs removes those vectors from subsequent search results.

- `tests/test_vector_stores.py::test_chroma_store_persists_across_instances`
  - Proves a Chroma store instance can reload and search data written by a previous instance under the tested persistent-client behavior.

## What This Means For Current RAG Readiness

The passing tests now prove that the backend query pipeline has the core RAG
wiring needed for real responses:

- Retrieval happens before answer generation.
- Retrieved chunk text and legal metadata are passed into `generate_with_context()`.
- The direct, non-RAG generation path remains available as a fallback.
- Retrieved chunks are not discarded after generation; they are reused for claim verification.
- Pipeline metadata distinguishes `rag` and `direct` generation.

The remaining practical work is operational and UI-facing:

- Ensure Ollama is running with the configured model for live generation.
- Ensure the intended BM25 and Chroma index artifacts exist before API startup.
- Rebuild and reload indices when new uploads should become searchable.
- Wire the Streamlit query UI to call the API instead of the placeholder local response path.
- Fix or intentionally remove the currently failing/missing tests listed in the scope limits.
