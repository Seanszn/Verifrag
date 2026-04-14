# LegalVerifiRAG System Summary

## Purpose

This document summarizes LegalVerifiRAG in a way that matches the development plans captured in this repository, not just the currently checked-in runtime state.

It reflects three things together:

- the target architecture described in the project summary, implementation spec, API notes, and tests
- the intended production direction: a hosted, server-based, enterprise-capable application
- the current implementation snapshot, so planned components are not confused with completed ones

## Primary Architectural Intent

The repository plans describe LegalVerifiRAG as a server-hosted legal verification platform with:

- a client application for end users
- a backend API service that owns authentication, conversations, query execution, and history
- retrieval, generation, and verification pipelines running behind that API
- deployment modes for local, cloud, and hybrid operation from one codebase
- a production path ranging from firm-local deployment to enterprise hosting

The planned architecture is not "pure Streamlit only." The intended shape is:

1. Clients run the client app.
2. The client app sends authenticated HTTP requests to the LegalVerifiRAG server.
3. The server executes retrieval, generation, and claim verification.
4. The server returns structured responses, evidence, and verification results.
5. The server persists users, sessions, conversations, and audit history.

This is consistent with:

- `LegalVerifiRAG_Project_Summary.md`
- `LegalVerifiRAG_Implementation_Spec (1).md`
- `legalverifirag/API_CALLS.md`
- `legalverifirag/tests/test_api_app.py`

## Intended Production Architecture

### Hosted Server Model

The intended application model is a hosted backend service, not a peer-to-peer client system.

Planned server responsibilities:

- expose API endpoints to client applications
- authenticate users and manage sessions
- execute the query pipeline
- manage retrieval indices and legal corpora
- persist conversations, messages, verified claims, and related records
- route LLM requests according to deployment mode and sensitivity rules

Planned client responsibilities:

- user login and logout
- query composition
- document upload
- viewing answers, citations, verification badges, and history
- calling backend endpoints over HTTP

### Enterprise Direction

The project docs explicitly plan for enterprise deployment, including:

- hosted server operation
- multi-user support
- authentication
- matter and client organization
- cost tracking and budget enforcement
- hybrid local/cloud routing for privacy-sensitive legal content
- Azure OpenAI as an enterprise LLM option
- Pinecone as a scalable production vector store option
- reverse-proxy fronting in production deployment diagrams

The planned progression in the repo is:

- laptop development
- capstone or small-firm deployment
- small-firm production
- enterprise production

### Deployment Modes

The design documents define three deployment modes from one codebase:

- `local`
  - local Ollama
  - local vector store
  - maximum privacy
- `cloud`
  - OpenAI, Anthropic, Gemini, or Azure OpenAI
  - cloud vector store such as Pinecone
  - highest answer quality and scalability
- `hybrid`
  - local models for privileged or uploaded content
  - cloud models for non-sensitive or higher-quality queries
  - recommended production direction in the implementation spec

## Intended Client-Server API Architecture

### Planned API Role

The repo plans consistently assume a backend API layer, specifically FastAPI, that sits between the client app and the retrieval or verification engine.

Planned backend role:

- receive requests from clients
- validate input
- enforce auth and session rules
- load conversation context
- run the pipeline
- store results
- return structured JSON responses

### Planned Endpoint Surface

Based on `API_CALLS.md`, README references, and tests, the intended API surface includes:

- `GET /health`
- `POST /api/auth/register`
- `POST /api/auth/login`
- `POST /api/auth/logout`
- `GET /api/conversations`
- `GET /api/conversations/{conversation_id}/messages`
- `POST /api/query`

The planned request/response pattern is:

1. Client authenticates and receives a bearer token.
2. Client sends token on protected routes.
3. Client submits a query to `/api/query`.
4. Server returns:
   - conversation metadata
   - user message
   - assistant message
   - pipeline metadata
   - claim verification details

### Network Model

The intended production model in the project summary is a normal hosted service on a server, typically inside a firm's network or behind a managed frontend.

The planning docs point to:

- Streamlit UI on one port
- FastAPI backend on another port
- services behind that layer
- production fronting through `nginx`

This repository does not describe `ngrok` as the intended production gateway. The design direction is a real hosted server deployment, not an ad hoc tunnel.

That means the expected connection model is:

- client app -> hosted API server endpoint

not:

- client app -> `ngrok` tunnel into a developer machine

## Intended End-to-End Flow

The planned system flow is:

1. User signs into the client app.
2. Client app calls the backend API.
3. Backend receives the query and any uploaded context.
4. Backend retrieves relevant legal passages from indexed corpora.
5. Backend generates an answer using the configured LLM backend.
6. Backend decomposes the answer into atomic legal claims.
7. Backend verifies each claim against retrieved evidence using the verification algorithms.
8. Backend assigns verdicts and confidence information.
9. Backend stores the interaction and returns the structured response to the client.
10. Client renders answer text, citations, history, and verification indicators.

This is the design-level flow the repo plans emphasize.

## Planned Component Map

### Client Layer

Planned role:

- client-facing UI
- auth forms
- query submission
- history view
- upload controls
- API consumption

Planned technologies:

- Streamlit in the current design documents
- backend HTTP client helpers in `src/client/api_client.py`

Expected inputs:

- username and password
- query text
- uploaded source files
- conversation selection

Expected outputs:

- API requests to the server
- rendered answer text
- verification badges or verdict labels
- history and conversation views

### API Layer

Planned role:

- primary server boundary for all client traffic

Planned technology:

- FastAPI

Expected inputs:

- JSON auth payloads
- bearer tokens
- query payloads
- conversation identifiers

Expected outputs:

- auth tokens
- conversation lists
- message history
- query results with pipeline metadata

### Pipeline Layer

Planned role:

- orchestration of retrieval, generation, verification, persistence, and response packaging

Expected inputs:

- authenticated user context
- query text
- optional uploaded document context
- deployment mode and routing config

Expected outputs:

- answer text
- retrieved evidence
- decomposed claims
- verification results
- final verdict labels
- cost and provider metadata where relevant

### Retrieval Layer

Planned role:

- dense plus sparse legal search
- reranking
- support for local or cloud vector storage

Planned stores:

- local FAISS or Chroma
- cloud Pinecone

Expected inputs:

- user query
- indexed legal chunks

Expected outputs:

- ranked `LegalChunk` evidence passages

### Generation Layer

Planned role:

- interchangeable local and cloud LLM backends
- privacy-aware routing
- cost-aware execution

Planned providers:

- Ollama
- OpenAI
- Azure OpenAI
- Anthropic
- Gemini
- hybrid router

Expected inputs:

- user query
- retrieved context
- sensitivity flags
- deployment mode

Expected outputs:

- generated legal response
- provider and model metadata
- optional cost metadata

### Verification Layer

Planned role:

- independent legal hallucination detection that runs regardless of generation provider

Planned algorithms:

- Algorithm 1: multi-evidence NLI aggregation with authority weighting
- Algorithm 2: legal claim decomposition
- Algorithm 3: contradiction detection
- Algorithm 4: final verdict classification
- Algorithm 5: citation verification

Expected inputs:

- assistant response
- retrieved legal evidence
- case and statute metadata

Expected outputs:

- claim-level support scores
- contradiction flags
- verdict labels
- citation validation results

### Storage Layer

Planned role:

- user accounts
- sessions
- conversations
- message history
- verified claims
- contradictions
- saved responses
- matter-level organization

Planned backends:

- SQLite for initial and small-firm scale
- migration path to PostgreSQL if needed

### Ingestion and Corpus Layer

Planned role:

- CourtListener downloads
- legal corpus synchronization
- PDF and DOCX ingestion
- user-upload parsing
- chunk preparation
- index builds

Expected inputs:

- CourtListener API data
- local user-uploaded documents

Expected outputs:

- raw JSONL corpora
- processed chunk JSONL
- retrieval indices

## Enterprise-Level Application Shape

If the repo is developed in line with its own plans, the enterprise-capable application should look like this:

### Server Side

- one hosted LegalVerifiRAG backend
- secured API endpoints for client applications
- centralized auth and session management
- centralized retrieval and verification execution
- centralized database and index management
- centralized logging, auditing, and cost tracking

### Client Side

- one or more client applications connecting to the server
- no direct access from client apps to raw indices or local model processes
- no need for clients to host the retrieval or verification stack themselves

### Production Infrastructure

Planned or implied production components in the repo docs:

- FastAPI backend service
- Streamlit client or frontend app
- reverse proxy such as `nginx`
- Ollama for local or privileged workloads
- cloud LLMs for enterprise or hybrid quality tiers
- local or cloud vector store
- database service

### Enterprise Concerns the Repo Already Plans For

- privileged-content routing
- cost visibility and budget limits
- configurable deployment modes
- scalable vector infrastructure
- enterprise provider support via Azure OpenAI
- matter-level organization
- multi-user operation

## Current Implementation Snapshot

The current checked-in code does not yet fully realize that target architecture.

### Present Today

- corpus download and sync scripts
- chunking and corpus preparation
- Chroma-backed vector storage
- BM25 retrieval
- deterministic claim decomposition
- NLI-based claim verification
- SQLite persistence for users, sessions, conversations, and messages
- a `QueryPipeline` orchestration module
- a Streamlit UI prototype
- client-side API helper scaffolding

### Planned But Not Fully Present in the Current Tree

Referenced in docs and tests, but not currently present as checked-in source:

- `src/api/main.py`
- supporting `src/api/*` modules
- a complete FastAPI server implementation
- cloud provider backends such as OpenAI, Anthropic, or Azure modules
- hybrid router and cost tracker modules
- Pinecone store implementation
- fully implemented contradiction, verdict, and citation verification modules
- the final client-to-server wiring in the current Streamlit app

### Important Distinction

The current codebase contains a partial implementation and a clearer architecture plan.

The right reading of this repository is:

- the intended application is a hosted client-server verification platform
- the currently checked-in code is an incomplete but directionally aligned implementation of that platform

## Recommended Interpretation for Ongoing Development

Going forward, this summary should be treated as the architectural target:

1. Build or restore the FastAPI backend under `src/api/`.
2. Make the client app consume that backend over authenticated endpoints.
3. Move all query execution to the server.
4. Keep retrieval and verification server-side.
5. Add enterprise-ready provider routing, storage, and deployment options behind configuration.
6. Preserve local, cloud, and hybrid modes from one codebase.

## Short Summary

LegalVerifiRAG is best understood as a planned enterprise-capable, server-hosted legal verification application with a client-server API architecture.

The intended production model is:

- client app sends authenticated requests to server endpoints
- server performs retrieval, generation, verification, persistence, and routing
- server returns structured legal answers and verification results

The current repository already contains many of the core pipeline pieces, but the full API layer and the final hosted client-server wiring remain to be completed.
