# Project Diagrams

These Mermaid diagrams are derived from the source in `src/storage/database.py`, `src/api/schemas.py`, `src/ingestion/document.py`, `src/api/uploads.py`, `src/indexing/index_builder.py`, `src/retrieval/user_uploads.py`, and `src/pipeline.py`.

## 1. Runtime Persistence ERD

```mermaid
erDiagram
    USERS {
        INTEGER id PK
        TEXT email UK
        TEXT username UK
        TEXT password_hash
        TEXT created_at
    }

    SESSIONS {
        INTEGER id PK
        INTEGER user_id FK
        TEXT token_hash UK
        TEXT created_at
        TEXT expires_at
    }

    CONVERSATIONS {
        INTEGER id PK
        INTEGER user_id FK
        TEXT title
        TEXT created_at
        TEXT updated_at
    }

    CONVERSATION_STATE {
        INTEGER conversation_id PK
        TEXT summary
        TEXT last_updated_at
    }

    INTERACTIONS {
        INTEGER id PK
        INTEGER conversation_id FK
        TEXT query
        TEXT response
        TEXT created_at
    }

    MESSAGES {
        INTEGER id PK
        INTEGER conversation_id FK
        INTEGER interaction_id FK
        TEXT role
        TEXT content
        TEXT created_at
        TEXT metadata_json
    }

    VERIFIED_CLAIMS {
        INTEGER id PK
        INTEGER interaction_id FK
        TEXT claim_text
        TEXT verdict
        REAL confidence
        TEXT source_citation
        TEXT created_at
        TEXT claim_id
        TEXT claim_type
        TEXT claim_source
        TEXT certainty
        TEXT doc_section
        TEXT span_json
        TEXT metadata_json
    }

    CONTRADICTIONS {
        INTEGER id PK
        INTEGER interaction_id FK
        TEXT chunk_i_id
        TEXT chunk_j_id
        REAL score
        TEXT created_at
        TEXT metadata_json
    }

    INTERACTION_CITATIONS {
        INTEGER id PK
        INTEGER interaction_id FK
        TEXT doc_id
        TEXT chunk_id
        TEXT source_label
        REAL score
        INTEGER used_in_prompt
        TEXT created_at
        TEXT metadata_json
    }

    CLAIM_CITATION_LINKS {
        INTEGER id PK
        INTEGER verified_claim_id FK
        INTEGER interaction_citation_id FK
        TEXT relationship
        REAL score
        TEXT created_at
        TEXT metadata_json
    }

    USERS ||--o{ SESSIONS : owns
    USERS ||--o{ CONVERSATIONS : owns
    CONVERSATIONS ||--|| CONVERSATION_STATE : tracks
    CONVERSATIONS ||--o{ INTERACTIONS : contains
    CONVERSATIONS ||--o{ MESSAGES : contains
    INTERACTIONS o|--o{ MESSAGES : groups
    INTERACTIONS ||--o{ VERIFIED_CLAIMS : stores
    INTERACTIONS ||--o{ CONTRADICTIONS : stores
    INTERACTIONS ||--o{ INTERACTION_CITATIONS : stores
    VERIFIED_CLAIMS ||--o{ CLAIM_CITATION_LINKS : links
    INTERACTION_CITATIONS ||--o{ CLAIM_CITATION_LINKS : supports_or_contradicts
```

## 2. API Contract ERD

Note: `citations`, `claim_citation_links`, and `contradictions` are open dict payloads reconstructed from stored metadata rather than strict Pydantic models.

```mermaid
erDiagram
    REGISTER_REQUEST {
        STRING username
        STRING password
    }

    LOGIN_REQUEST {
        STRING username
        STRING password
    }

    USER_RESPONSE {
        INTEGER id
        STRING username
        STRING email
        STRING created_at
    }

    AUTH_RESPONSE {
        STRING token
    }

    CONVERSATION_CREATE_REQUEST {
        STRING title
    }

    CONVERSATION_SUMMARY {
        INTEGER id
        INTEGER user_id
        STRING title
        STRING created_at
        STRING updated_at
    }

    MESSAGE_RESPONSE {
        INTEGER id
        INTEGER conversation_id
        INTEGER interaction_id
        STRING role
        STRING content
        STRING created_at
        STRING metadata_json
    }

    INTERACTION_RESPONSE {
        INTEGER id
        INTEGER conversation_id
        STRING query
        STRING response
        STRING created_at
    }

    CLAIM_SPAN_RESPONSE {
        STRING doc_id
        INTEGER para_id
        INTEGER sent_id
        INTEGER start_char
        INTEGER end_char
        STRING text
    }

    CLAIM_EVIDENCE_RESPONSE {
        STRING relationship
        REAL score
        STRING chunk_id
        STRING doc_id
        STRING source_label
        JSON citation
    }

    CLAIM_ANNOTATION_RESPONSE {
        STRING support_level
        STRING explanation
    }

    CLAIM_RESPONSE {
        STRING claim_id
        STRING text
        STRING claim_type
        STRING source
        STRING certainty
        STRING doc_section
        JSON span
        JSON verification
    }

    PIPELINE_RESPONSE {
        INTEGER claim_count
    }

    QUERY_REQUEST {
        STRING query
        INTEGER conversation_id
    }

    QUERY_RESPONSE {
        STRING response_envelope
    }

    INTERACTION_DETAIL_RESPONSE {
        STRING interaction_bundle
    }

    CITATION_PAYLOAD {
        JSON payload
    }

    CLAIM_CITATION_LINK_PAYLOAD {
        JSON payload
    }

    CONTRADICTION_PAYLOAD {
        JSON payload
    }

    AUTH_RESPONSE ||--|| USER_RESPONSE : includes
    REGISTER_REQUEST ||--|| AUTH_RESPONSE : returns
    LOGIN_REQUEST ||--|| AUTH_RESPONSE : returns

    QUERY_REQUEST o|--|| CONVERSATION_SUMMARY : targets_existing_conversation

    QUERY_RESPONSE ||--|| CONVERSATION_SUMMARY : includes
    QUERY_RESPONSE ||--|| INTERACTION_RESPONSE : includes
    QUERY_RESPONSE ||--o{ MESSAGE_RESPONSE : returns_messages
    QUERY_RESPONSE ||--|| PIPELINE_RESPONSE : includes

    PIPELINE_RESPONSE ||--o{ CLAIM_RESPONSE : contains
    CLAIM_RESPONSE o|--|| CLAIM_ANNOTATION_RESPONSE : exposes
    CLAIM_RESPONSE o|--|| CLAIM_SPAN_RESPONSE : span_details
    CLAIM_RESPONSE ||--o{ CLAIM_EVIDENCE_RESPONSE : linked_evidence

    INTERACTION_DETAIL_RESPONSE ||--|| INTERACTION_RESPONSE : includes
    INTERACTION_DETAIL_RESPONSE ||--o{ CLAIM_RESPONSE : includes
    INTERACTION_DETAIL_RESPONSE ||--o{ CITATION_PAYLOAD : includes
    INTERACTION_DETAIL_RESPONSE ||--o{ CLAIM_CITATION_LINK_PAYLOAD : includes
    INTERACTION_DETAIL_RESPONSE ||--o{ CONTRADICTION_PAYLOAD : includes
```

## 3. Corpus, Upload, And Index Artifact ERD

This ERD models the file-backed schemas used for corpus preparation and user-upload overlays.

```mermaid
erDiagram
    LEGAL_DOCUMENT {
        STRING id PK
        STRING doc_type
        TEXT full_text
        STRING case_name
        STRING citation
        STRING court
        STRING court_level
        DATE date_decided
        INTEGER title
        STRING section
        STRING source_file
        BOOLEAN is_privileged
    }

    LEGAL_CHUNK {
        STRING id PK
        STRING doc_id FK
        TEXT text
        INTEGER chunk_index
        STRING doc_type
        STRING case_name
        STRING court
        STRING court_level
        STRING citation
        DATE date_decided
        INTEGER title
        STRING section
        STRING source_file
    }

    RAW_JSONL_ROW {
        STRING id PK
        STRING doc_type
        TEXT full_text
        STRING case_name
        STRING citation
        STRING court
        STRING court_level
        STRING date_decided
        STRING source_file
        BOOLEAN is_privileged
        STRING uploaded_at
    }

    PROCESSED_CHUNK_ROW {
        STRING id PK
        STRING doc_id FK
        TEXT text
        INTEGER chunk_index
        STRING doc_type
        STRING case_name
        STRING court
        STRING court_level
        STRING citation
        STRING date_decided
        INTEGER title
        STRING section
        STRING source_file
    }

    BM25_ARTIFACT {
        STRING bm25_path PK
        INTEGER chunk_count
        STRING serialization_format
    }

    CHROMA_COLLECTION {
        STRING chroma_path PK
        STRING collection_name
        STRING distance_metric
        INTEGER embedding_dim
    }

    INDEX_SUMMARY {
        STRING summary_path PK
        STRING processed_dir
        INTEGER chunk_count
        STRING embedding_model
        STRING built_at
    }

    USER_UPLOAD_BATCH {
        INTEGER user_id
        INTEGER conversation_id
        BOOLEAN is_privileged
        INTEGER files_uploaded
        INTEGER documents_upserted
        INTEGER chunks_upserted
    }

    UPLOAD_FILE_SUMMARY {
        STRING filename
        STRING content_type
        STRING document_id FK
        INTEGER size_bytes
        INTEGER chunk_count
        BOOLEAN is_privileged
    }

    LEGAL_DOCUMENT ||--|| RAW_JSONL_ROW : serialized_as
    LEGAL_DOCUMENT ||--o{ LEGAL_CHUNK : splits_into
    LEGAL_CHUNK ||--|| PROCESSED_CHUNK_ROW : serialized_as
    PROCESSED_CHUNK_ROW }o--|| BM25_ARTIFACT : indexed_into
    PROCESSED_CHUNK_ROW }o--|| CHROMA_COLLECTION : embedded_into
    BM25_ARTIFACT ||--|| INDEX_SUMMARY : described_by
    CHROMA_COLLECTION ||--|| INDEX_SUMMARY : described_by
    USER_UPLOAD_BATCH ||--o{ UPLOAD_FILE_SUMMARY : returns
    UPLOAD_FILE_SUMMARY }o--|| LEGAL_DOCUMENT : materializes
```

## 4. General Architecture

```mermaid
flowchart TB
    user[User]
    client[Streamlit Client<br/>src/client/ui.py]

    subgraph api[FastAPI Server]
        auth[Auth Routes<br/>/api/auth/*]
        conv[Conversation Routes<br/>/api/conversations/*]
        query[Query Route<br/>/api/query]
        uploads[Upload Route<br/>/api/uploads]
        deps[Shared Dependencies<br/>Database + QueryPipeline singletons]
        pipeline[QueryPipeline<br/>src/pipeline.py]
    end

    subgraph runtime[Runtime Services]
        sqlite[(SQLite<br/>data/legalverifirag.db)]
        ollama[Ollama HTTP API]
        verifier[Claim Decomposer + NLI Verifier]
        contract[Claim Contract Normalizer]
        hybrid[HybridRetriever]
        publicidx[Public Indexes<br/>BM25 + Chroma]
        useridx[User Upload Indexes<br/>per-user BM25 + Chroma]
    end

    subgraph uploads_data[User Upload Workspace]
        files[Stored Originals<br/>data/uploads/user_{id}/files]
        rawjsonl[Raw Upload JSONL<br/>data/uploads/user_{id}/raw]
        processedjsonl[Processed Chunk JSONL<br/>data/uploads/user_{id}/processed]
        uploadbuild[Index Builder<br/>build_user_upload_indices]
    end

    subgraph offline[Offline Corpus Build]
        courtlistener[CourtListener API]
        corpusbuilder[CorpusBuilder]
        rawcorpus[Raw Corpus JSONL<br/>data/raw]
        preparer[prepare_corpus]
        processedcorpus[Processed Chunk JSONL<br/>data/processed]
        indexbuilder[build_indices]
        indexsummary[Index Summary<br/>data/index/*]
    end

    user --> client
    client -->|HTTP| auth
    client -->|HTTP| conv
    client -->|HTTP| query
    client -->|multipart upload| uploads

    auth --> deps
    conv --> deps
    query --> deps
    uploads --> deps

    auth --> sqlite
    conv --> sqlite
    query --> pipeline
    pipeline --> sqlite
    pipeline --> hybrid
    pipeline --> ollama
    pipeline --> verifier
    verifier --> contract
    pipeline --> contract

    hybrid --> publicidx
    hybrid --> useridx

    uploads --> files
    uploads --> rawjsonl
    uploads --> processedjsonl
    uploads --> uploadbuild
    uploadbuild --> useridx

    courtlistener --> corpusbuilder
    corpusbuilder --> rawcorpus
    rawcorpus --> preparer
    preparer --> processedcorpus
    processedcorpus --> indexbuilder
    indexbuilder --> publicidx
    indexbuilder --> indexsummary
```
