# LegalVerifiRAG API Implementation Plan

## Purpose

This document lays out a practical way to create the full `src/api/` folder and connect it to the existing `src/` modules.

It is written as an implementation plan in the order that makes the most sense to build it.

The goal is to move the project from:

- a partly local Streamlit prototype

to:

- a proper client-server application where the client app sends authenticated requests to a hosted backend and receives structured responses

## End State

By the end of this work, the application should look like this:

1. The client app logs in through backend endpoints.
2. The client app submits queries to the backend.
3. The backend owns:
   - authentication
   - sessions
   - conversations
   - query execution
   - retrieval
   - verification
   - upload handling
4. The backend returns structured JSON responses.
5. The client app becomes a thin UI layer instead of directly touching the database or local prototype state.

## Core Rule

When building the API layer, the main architectural rule should be:

- `src/client/` should call `src/api/`
- `src/api/` should call the existing business logic in the rest of `src/`
- `src/client/` should not directly call database, auth hashing, or query pipeline internals

That means:

- no direct `Database()` usage in the Streamlit UI
- no local password hashing in the UI
- no placeholder local query responses in the UI
- no client-owned authoritative upload handling

## Existing Modules You Should Reuse

The API folder should connect to these existing modules first:

- `src/storage/database.py`
  - user, session, conversation, message persistence
- `src/auth/local_auth.py`
  - password hashing and verification
- `src/pipeline.py`
  - query orchestration
- `src/verification/claim_decomposer.py`
  - already used indirectly by the pipeline
- `src/verification/nli_verifier.py`
  - already used indirectly by the pipeline
- `src/retrieval/hybrid_retriever.py`
  - already used indirectly by the pipeline
- `src/generation/ollama_backend.py`
  - already used indirectly by the pipeline
- `src/ingestion/pdf_parser.py`
  - should be used by upload endpoints later
- `src/client/api_client.py`
  - should become the client’s transport layer into the backend

## Recommended File Layout

Create this structure under `src/api/`:

```text
src/
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── dependencies.py
│   ├── schemas.py
│   ├── errors.py
│   ├── middleware.py
│   ├── health.py
│   ├── auth.py
│   ├── query.py
│   ├── conversations.py
│   ├── uploads.py
│   └── services.py
```

### File Responsibilities

- `main.py`
  - create the FastAPI app
  - register routers
  - install middleware
  - initialize shared services at startup

- `dependencies.py`
  - shared `Database`
  - shared `QueryPipeline`
  - current-user dependency from bearer token

- `schemas.py`
  - all request and response models

- `errors.py`
  - shared API exception helpers and handlers

- `middleware.py`
  - CORS
  - request logging
  - security headers

- `health.py`
  - `/health`

- `auth.py`
  - register
  - login
  - logout

- `query.py`
  - `/api/query`

- `conversations.py`
  - list conversations
  - get messages

- `uploads.py`
  - upload files to server
  - parse and ingest them

- `services.py`
  - optional thin wrappers around common API-level operations
  - useful if route files start becoming too crowded

## Best Implementation Order

This is the recommended order to implement everything.

Do not start with uploads first. Do not start with the client first. Get the backend skeleton working before wiring the UI.

---

## Phase 1: Create The API Skeleton

### Step 1. Create `src/api/__init__.py`

Purpose:

- mark the folder as a package

This file can be empty.

### Step 2. Create `src/api/main.py`

Purpose:

- this is the assembly point for the whole backend

It should:

- create the FastAPI app
- include routers
- register middleware
- expose `/health`
- initialize the shared database and pipeline on startup

At this stage, keep it minimal.

### Step 3. Create `src/api/dependencies.py`

Purpose:

- centralize service creation and dependency injection

Start by wiring:

- `Database`
- `QueryPipeline`
- current-user auth dependency

This file is important because it prevents every route from manually rebuilding the DB or pipeline.

Suggested responsibilities:

- create one shared `Database`
- call `initialize()`
- create one shared `QueryPipeline`
- implement `get_db()`
- implement `get_pipeline()`
- implement `get_current_user()`

### Step 4. Create `src/api/health.py`

Purpose:

- simple readiness endpoint

Implement:

- `GET /health`

It should return something minimal like:

```json
{ "status": "ok" }
```

This is useful immediately for checking whether the backend is up before you start wiring auth or queries.

---

## Phase 2: Define Request And Response Models

### Step 5. Create `src/api/schemas.py`

Purpose:

- define all API payloads in one place

Start with:

- `RegisterRequest`
- `LoginRequest`
- `AuthResponse`
- `QueryRequest`
- `QueryResponse`

Then add:

- `ConversationSummary`
- `MessageResponse`
- `UploadResponse`

Keep the first pass simple. The point is to stabilize the interface between client and server.

Why this comes early:

- route implementation is much easier when the payload shape is already explicit
- the client app can later be rewritten directly against these schemas

---

## Phase 3: Move Authentication To The Server

### Step 6. Create `src/api/auth.py`

Purpose:

- move all register, login, and logout logic into the backend

Implement first:

- `POST /api/auth/register`
- `POST /api/auth/login`
- `POST /api/auth/logout`

Connect this file to:

- `src/storage/database.py`
- `src/auth/local_auth.py`

Important rule:

- use `src/auth/local_auth.py`
- do not copy the current `bcrypt` logic from the Streamlit UI

Why:

- `local_auth.py` is the shared server-side auth implementation
- the current UI auth path is inconsistent with it
- the API must become the single source of truth

### Auth behavior to implement

#### Register

Flow:

1. validate username and password
2. check whether username already exists
3. hash password with `hash_password()`
4. create user in DB
5. create session token
6. return `{ token, user }`

#### Login

Flow:

1. fetch user by username
2. verify password using `verify_password()`
3. create session token
4. return `{ token, user }`

#### Logout

Flow:

1. read bearer token
2. invalidate the session
3. return `204 No Content`

### Step 7. Implement `get_current_user()` in `dependencies.py`

Purpose:

- let protected routes depend on authenticated user context

This should:

1. read the `Authorization` header
2. require `Bearer <token>`
3. look up the user via `Database.get_user_for_token()`
4. reject invalid or expired tokens with `401`

Once this is done, all protected routes can use the same dependency.

---

## Phase 4: Expose Conversation History Through The API

### Step 8. Create `src/api/conversations.py`

Purpose:

- expose persisted history to clients

Implement:

- `GET /api/conversations`
- `GET /api/conversations/{conversation_id}/messages`

Connect this file to:

- `src/storage/database.py`

Behavior:

- both endpoints must require auth
- both endpoints must only return records owned by the current user

This phase should come before query submission because it gives you a simple, testable protected route surface first.

---

## Phase 5: Expose Query Execution Through The API

### Step 9. Create `src/api/query.py`

Purpose:

- expose the actual query pipeline

Implement:

- `POST /api/query`

Connect this file to:

- `src/pipeline.py`
- `src/api/dependencies.py`

The route should:

1. require auth
2. accept `query` and optional `conversation_id`
3. call `QueryPipeline.run(...)`
4. return the result directly as structured JSON

This is the key backend endpoint because it becomes the server-owned execution boundary.

### Important note on pipeline behavior

The current `QueryPipeline` already handles:

- conversation creation or reuse
- storing the user message
- generating an assistant response
- decomposing the response into claims
- verifying claims when retrieval is available
- storing the assistant message

That means the API route itself should stay thin.

Do not reimplement pipeline logic in the route.

---

## Phase 6: Add Error Handling And Middleware

### Step 10. Create `src/api/errors.py`

Purpose:

- normalize API error responses

Add:

- helpers for `401`, `404`, `409`, and validation-style domain errors
- optional custom exception classes if needed later

This is not strictly required on day one, but it helps keep route files clean.

### Step 11. Create `src/api/middleware.py`

Purpose:

- centralize cross-cutting HTTP behavior

Start with:

- CORS
- request logging

Then optionally add:

- security headers
- correlation IDs
- response timing logs

If the client app and backend are hosted separately, CORS matters immediately.

---

## Phase 7: Add Upload Endpoints

### Step 12. Create `src/api/uploads.py`

Purpose:

- move document handling from the client into the server

This is where "server-ingested documents" begins.

Implement first:

- `POST /api/uploads`

Do not try to build final persistent indexing on the first pass. Start with session-aware or temp-server ingestion.

### What server-ingested should mean

Instead of this flow:

1. client receives file
2. client writes file locally
3. client only references filename in UI

You want this flow:

1. client uploads file to API
2. server stores it temporarily or persistently
3. server parses it using `src/ingestion/pdf_parser.py` or other ingestion code
4. server converts it into `LegalDocument`
5. server either:
   - keeps it in temporary session retrieval context
   - or writes it into raw and processed corpora
6. later query execution can actually use that document as evidence

### Best first upload approach

For the first implementation, use one of these:

#### Option A: Session-only upload handling

- easiest
- good first milestone

Flow:

- upload file
- server parses it
- server stores parsed text in a temp session store
- query route can include those temporary chunks later

#### Option B: Persistent corpus ingestion

- stronger long-term solution
- more engineering

Flow:

- upload file
- parse it
- write raw JSONL
- chunk it
- update processed artifacts
- rebuild or update indices

For tomorrow, start with Option A unless you already know you need permanent corpus integration immediately.

---

## Phase 8: Wire The Client To The API

### Step 13. Rewrite `src/client/ui.py` to stop using direct local logic

Purpose:

- convert the client into a real frontend for the API

You should remove or replace these behaviors:

- direct `Database()` construction
- local registration and login logic
- local bcrypt handling
- local conversation ownership
- placeholder query generation

The client should instead:

- call auth endpoints
- store only the bearer token and lightweight UI state
- load conversations through the API
- submit queries through the API
- render server responses

### Step 14. Expand `src/client/api_client.py`

Purpose:

- make it the single transport layer between UI and backend

Add helpers for:

- `register(...)`
- `login(...)`
- `logout(...)`
- `submit_query(...)`
- `upload_files(...)`

This keeps HTTP code out of the Streamlit page logic.

---

## Phase 9: Add Run Scripts For The API

### Step 15. Add `scripts/run_api.ps1` and `scripts/run_api.sh`

Purpose:

- run the backend as a separate process

These scripts should start:

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

For a real hosted server deployment later, this host value will likely change from loopback to a server-bound interface or sit behind a reverse proxy.

### Step 16. Keep `run_local` for the client UI only

Purpose:

- separate backend startup from frontend startup

The server and client must become different runtime processes.

---

## Phase 10: Align Tests With The API

### Step 17. Make `tests/test_api_app.py` pass

Purpose:

- this becomes your first backend integration milestone

This test already points to the right architectural shape:

- register
- query
- read conversation messages

Use it as the initial proof that the backend is working.

### Step 18. Add follow-up tests

After the basics work, add:

- auth failure tests
- unauthorized route tests
- invalid conversation access tests
- health endpoint tests
- upload tests

---

## Practical Tomorrow Checklist

If you are doing this tomorrow, the clean order is:

1. Create `src/api/__init__.py`
2. Create `src/api/main.py`
3. Create `src/api/dependencies.py`
4. Create `src/api/schemas.py`
5. Create `src/api/health.py`
6. Create `src/api/auth.py`
7. Create `src/api/conversations.py`
8. Create `src/api/query.py`
9. Add `scripts/run_api.ps1`
10. Start backend and hit `/health`
11. Make `tests/test_api_app.py` pass
12. Only then start rewriting `src/client/ui.py`
13. After client auth and query work, add `src/api/uploads.py`

That order keeps the work incremental and testable.

## Suggested Endpoint Order

Implement endpoints in this order:

1. `GET /health`
2. `POST /api/auth/register`
3. `POST /api/auth/login`
4. `POST /api/auth/logout`
5. `GET /api/conversations`
6. `GET /api/conversations/{conversation_id}/messages`
7. `POST /api/query`
8. `POST /api/uploads`

This order matters because:

- health proves the app boots
- auth proves protected routing works
- conversations prove user-scoped DB reads work
- query proves pipeline integration works
- uploads should come after the server execution path is stable

## What To Avoid

Do not do these things:

- do not let the client keep directly touching SQLite
- do not keep two auth systems alive long-term
- do not duplicate query logic in the route and pipeline
- do not start by building the most complex upload/indexing behavior first
- do not keep the placeholder response path once `/api/query` exists
- do not let UI session state become the source of truth for persisted conversations

## Design Decisions You Should Make Early

There are three decisions worth making before you get too deep:

### 1. Should the backend remain local-network hosted first, or public-server hosted first?

Recommended first answer:

- local-network hosted first

Why:

- simpler
- matches the project’s local and hybrid emphasis
- easier for early testing

### 2. Should uploaded files be session-only first or persistent first?

Recommended first answer:

- session-only first

Why:

- much simpler
- enough to prove the client-server ingestion model

### 3. Should query generation remain answer-first or move to retrieval-first now?

Recommended first answer:

- keep the current pipeline first
- expose it through the API
- then refactor the pipeline once the API surface is stable

Why:

- reduces moving parts during the first backend build

## Final Recommended Sequence

The shortest sensible route is:

1. build the API shell
2. centralize dependencies
3. move auth to the server
4. expose conversations
5. expose `/api/query`
6. make tests pass
7. rewire the Streamlit UI to call the API
8. add upload ingestion
9. then improve the pipeline and deployment behavior

## Short Summary

Tomorrow, do not try to build everything at once.

Build the backend in this order:

1. app skeleton
2. dependencies
3. schemas
4. auth routes
5. conversation routes
6. query route
7. run scripts
8. client rewiring
9. uploads

Use the API layer as a thin boundary that delegates into the existing `src/` modules, especially:

- `database.py`
- `local_auth.py`
- `pipeline.py`
- `pdf_parser.py`

That will get the project aligned with the intended hosted client-server architecture without forcing a full rewrite on day one.
