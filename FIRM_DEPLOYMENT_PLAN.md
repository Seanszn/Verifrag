# LegalVerifiRAG Firm Deployment Plan

This repository is closest to a central-server architecture where one firm-hosted
backend owns storage, retrieval, generation, and verification, and paralegals run
only the Streamlit UI on their own devices.

The deployment shape should be:

1. Firm server runs the LegalVerifiRAG API.
2. Firm server owns SQLite or a replacement database, Chroma/BM25 index files,
   uploaded documents, and the LLM connection.
3. Paralegal laptops run the Streamlit UI only.
4. The Streamlit UI talks to the firm API over HTTPS.

## Minimal Server Containerization

The added container files are intentionally server-only:

- `Dockerfile.server`
- `docker-compose.server.yml`

This is deliberate. The current codebase is not ready for a "single image runs
everything everywhere" deployment because:

- `src/api/main.py` is empty today
- the UI still performs local auth and local file persistence
- the checked-in API docs describe endpoints that are not currently implemented
- local Ollama remains a separate runtime concern

The server container should therefore be treated as the packaging target for the
intended backend, not proof that the current backend is complete.

## Recommended Firm Deployment Vision

### Central Server Responsibilities

The server should own:

- user authentication and session issuance
- conversation history and audit history
- uploaded document ingestion
- corpus preparation and indexing
- retrieval, generation, and verification
- model routing to Ollama or a hosted provider

The server should persist:

- database files under `data/`
- Chroma and BM25 artifacts under `data/index/`
- uploaded source files in a server-side upload directory
- logs and operational metrics

### Paralegal Device Responsibilities

Each paralegal device should own only:

- the Streamlit UI process
- local session state for the current browser session
- the API base URL for the firm server

Each paralegal device should not own:

- a local database
- local document ingestion storage
- local Chroma/BM25 indices
- local generation or verification models
- direct Ollama access

## What Needs To Change In The UI And Client

### 1. Remove Database Access From Streamlit

Current problem:

- `src/client/ui.py` imports `Database`
- registration and login are handled directly in the UI
- the UI creates and reads local SQLite state

Required change:

- remove direct `Database` usage from the Streamlit client
- remove local password hashing from the UI
- replace register/login/logout flows with HTTP calls to the backend API

The client should only store:

- bearer token
- current user profile returned by the server
- selected conversation id
- transient page state

### 2. Wire The UI To `src/client/api_client.py`

Current problem:

- `src/client/api_client.py` exists
- `src/client/ui.py` does not use it

Required change:

- use `api_client.py` for login, logout, conversation list, message history, and query submission
- add explicit API helpers for auth routes and query submission
- handle `401` by clearing local auth state and redirecting to login
- surface backend validation errors directly in Streamlit

### 3. Replace Local Upload Persistence With API Uploads

Current problem:

- `process_uploaded_files()` writes files into local `temp_uploads/` on the paralegal machine
- uploaded files are tracked only in Streamlit session state

Required change:

- replace local file writes with multipart upload requests to the backend
- server stores files in a managed upload area
- server associates uploaded files with a user, matter, or conversation
- UI keeps only server-returned metadata such as file name, upload id, size, and status

Recommended API additions:

- `POST /api/uploads`
- `GET /api/uploads`
- `DELETE /api/uploads/{upload_id}`

### 4. Replace Local Session Transcript State With Server Conversations

Current problem:

- Streamlit stores conversations in `st.session_state["sessions"]`
- generated chat messages are local placeholders

Required change:

- fetch conversations from the backend after login
- fetch messages for the selected conversation
- submit queries to the backend
- render backend-returned assistant messages and pipeline metadata

The UI should stop inventing local conversation objects and instead treat the
backend as the source of truth.

### 5. Replace Placeholder Answers With Real Query Calls

Current problem:

- `build_placeholder_response()` still generates fake output

Required change:

- remove placeholder generation
- call `POST /api/query`
- render the returned response, citations, retrieved evidence, and claim verification results

### 6. Add Backend Health And Connectivity Handling

Because the UI runs on paralegal devices and the server runs centrally, the UI
needs explicit network behavior:

- show backend health or connection status
- fail cleanly if the server is unavailable
- retry idempotent reads where appropriate
- show a firm-friendly error when login or query submission cannot reach the API

### 7. Treat Streamlit As A Thin Client

The Streamlit app should be presentation-only. It can still own:

- layout
- auth forms
- upload forms
- chat transcript rendering
- verification badges and evidence display

It should not own:

- auth logic
- storage logic
- ingestion logic
- retrieval logic
- generation logic
- verification logic

## What Needs To Change On The Server

The client deployment vision depends on a real backend implementation. That means:

- implement `src/api/main.py`
- add auth dependencies and bearer-token validation
- expose conversation and query endpoints documented in `API_CALLS.md`
- add upload endpoints
- connect query routes to `QueryPipeline`
- normalize file/index paths so runtime defaults match built artifacts

The current repo still has a gap here: tests and docs assume the API exists, but
the entrypoint file is empty.

## Practical Rollout Order

1. Implement the FastAPI app and auth/query routes.
2. Refactor the Streamlit UI into a pure HTTP client.
3. Add upload endpoints and remove local upload persistence.
4. Normalize index/bootstrap behavior on the server.
5. Run the API on the firm server with Docker Compose.
6. Point paralegal Streamlit clients at the firm API URL.
7. Put the API behind TLS and a reverse proxy before wider internal rollout.

## Recommendation On Docker

Docker makes sense for the firm-hosted server.

Docker does not meaningfully simplify the paralegal-side Streamlit UI unless you
intend to distribute the UI as a local container, which is usually more friction
than value for non-technical users. For paralegals, a small local Python install
or packaged desktop distribution is usually cleaner than "run Docker Desktop and
start a container."

So the right split is:

- central server: Dockerized
- paralegal client: thin Streamlit app, not necessarily Dockerized

## Near-Term Architecture Choice

If the firm deployment vision is the goal, stop investing in the current
prototype pattern where Streamlit acts as both UI and local application runtime.

The target should be:

- Streamlit = client
- FastAPI = application server
- data/index/model access = server-side only
