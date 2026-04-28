# LegalVerifiRAG

LegalVerifiRAG is a local-first legal retrieval and verification system. It answers legal questions from an indexed corpus, decomposes generated answers into atomic claims, verifies each claim against retrieved evidence, and displays support status with citations and evidence links.

The project is built around a FastAPI backend, a Streamlit client, local retrieval indices, local Ollama generation, and local NLI-based claim verification.

## Current Functionality

- User registration, login, logout, and bearer-token sessions.
- Conversation history with messages, interactions, verified claims, citations, and claim-citation links.
- Retrieval-grounded legal answers from a public corpus.
- Hybrid retrieval with BM25 sparse search, ChromaDB dense vector search, and Reciprocal Rank Fusion.
- Conservative query expansion for broad legal-research queries, with named-case and citation queries kept exact.
- Local Ollama answer generation.
- Deterministic claim decomposition with response-span tracking.
- Batched NLI claim verification with calibrated verdicts.
- Heuristic verification mode and optional fallback when live NLI fails.
- Inline support rendering in the Streamlit UI.
- Claim-level evidence inspection, supporting and contradicting case references, and answer-block support metadata.
- Upload ingestion for PDF, DOCX, Markdown, and text files.
- Per-user upload retrieval overlays that can be included in queries.
- Server-side persistence in SQLite.
- Docker deployment for API, client, and optional Ollama.
- Local setup and run scripts for Windows PowerShell and Linux/macOS shells.

## Architecture

```text
Browser
  -> Streamlit client on 8501
  -> FastAPI backend on 8000
  -> QueryPipeline
  -> Hybrid retrieval, Ollama generation, claim decomposition, NLI verification
  -> SQLite, ChromaDB, BM25, JSONL corpus files, user upload indices
```

The Streamlit app is a client. The backend owns authentication, uploads, query execution, persistence, retrieval, generation, and verification.

This README is the canonical operations guide for the current project layout.

## Repository Layout

```text
legalverifirag/
|-- src/
|   |-- api/            FastAPI backend routes and dependencies
|   |-- auth/           Password hashing helpers
|   |-- client/         Streamlit UI and API client
|   |-- generation/     Ollama backend and prompt templates
|   |-- indexing/       BM25, ChromaDB, embeddings, index discovery
|   |-- ingestion/      Corpus, PDF, upload, and chunk processing
|   |-- retrieval/      Hybrid public and user-upload retrievers
|   |-- storage/        SQLite schema and persistence
|   |-- verification/   Claim decomposition, NLI, verdicts, frontend contract
|   |-- pipeline.py     Server-side query orchestration
|   `-- app.py          Streamlit entrypoint
|-- scripts/            Setup, corpus, indexing, probes, and evaluation scripts
|-- data/
|   |-- raw/            Downloaded corpus JSONL
|   |-- processed/      Chunked corpus JSONL
|   |-- index/          BM25, ChromaDB, index summary
|   |-- uploads/        Per-user uploaded documents and indices
|   `-- eval/           Evaluation datasets
|-- assets/             UI assets
|-- tests/              Unit and integration tests
|-- docker-compose.yml
|-- docker-compose.server.yml
|-- Dockerfile.server
`-- Dockerfile.client
```

## Requirements

For native local runs:

- Python 3.10 or newer; Python 3.12 is used by the Docker images.
- Ollama installed separately.
- The configured Ollama model pulled locally, default `llama3.2:3b`.
- Enough disk space for Python packages, Hugging Face models, ChromaDB, and corpus data.
- Optional CourtListener API token for corpus downloads.

For Docker runs:

- Docker Engine and Docker Compose v2.
- Ollama on the host, in the optional Compose profile, or on another reachable server.
- Network access during image build if you want Docker to prefetch the configured NLI model.

## Configuration

Create a local `.env` from the example:

```powershell
Copy-Item .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

Important settings:

```env
DEPLOYMENT_MODE=local
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2:3b
OLLAMA_HOST=http://localhost:11434
LLM_MAX_TOKENS=1024
LLM_REQUEST_TIMEOUT_SECONDS=150
OLLAMA_NUM_CTX=4096

API_HOST=127.0.0.1
API_PORT=8000
API_BASE_URL=http://127.0.0.1:8000
API_CONNECT_TIMEOUT_SECONDS=10
API_REQUEST_TIMEOUT_SECONDS=30
API_QUERY_TIMEOUT_SECONDS=180

DATABASE_PATH=data/legalverifirag.db
CHROMA_PATH=data/index/chroma
CHROMA_COLLECTION=legal_chunks
QUERY_EXPANSION_MODE=hybrid
QUERY_EXPANSION_MAX_VARIANTS=5
QUERY_EXPANSION_MAX_TERMS=16

ENABLE_VERIFICATION=true
VERIFICATION_VERIFIER_MODE=live
VERIFICATION_FALLBACK_TO_HEURISTIC=true
NLI_MODEL=MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
NLI_DEVICE=cuda
NLI_BATCH_SIZE=1
NLI_MAX_LENGTH=384
NLI_DTYPE=auto
NLI_UNLOAD_AFTER_REQUEST=false

COURTLISTENER_TOKEN=
```

Notes:

- `VERIFICATION_VERIFIER_MODE=heuristic` avoids loading the live NLI model.
- `ENABLE_VERIFICATION=false` is useful for faster generation-only smoke tests.
- Set `NLI_DEVICE=cpu` on machines without CUDA.
- Set `NLI_UNLOAD_AFTER_REQUEST=true` when GPU memory is tight and Ollama needs memory reclaimed after verification.
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_LOCAL_FILES_ONLY=1` should only be enabled after required Hugging Face models are already cached.
- `QUERY_EXPANSION_MODE=hybrid` expands broad research-lead questions such as requests to find cases or authorities. Named-case and citation-specific questions are not expanded.

## Native Local Deployment

Run commands from the `legalverifirag/` directory.

### Windows PowerShell

```powershell
.\scripts\setup_local.ps1
ollama serve
ollama pull llama3.2:3b
.\scripts\run_api.ps1
```

Open a second terminal:

```powershell
.\scripts\run_local.ps1
```

Then open:

```text
http://127.0.0.1:8501
```

### Linux/macOS

```bash
chmod +x scripts/setup_local.sh scripts/run_api.sh scripts/run_local.sh
./scripts/setup_local.sh
ollama serve
ollama pull llama3.2:3b
./scripts/run_api.sh
```

Open a second terminal:

```bash
./scripts/run_local.sh
```

Then open:

```text
http://127.0.0.1:8501
```

### Manual Native Commands

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env
ollama pull llama3.2:3b
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
source venv/bin/activate
python -m streamlit run src/app.py
```

On Windows PowerShell, activate with:

```powershell
.\venv\Scripts\Activate.ps1
```

## Building A Retrieval Corpus

The app can run without a retrieval index, but answers and verification are strongest after building one.

Download a small SCOTUS test corpus:

```bash
python scripts/download_corpus.py --scotus --limit 10
```

Prepare raw documents into chunks:

```bash
python scripts/prepare_corpus.py
```

Build BM25 and ChromaDB indices:

```bash
python scripts/build_index.py
```

Full or incremental sync examples:

```bash
python scripts/download_corpus.py --all
python scripts/download_corpus.py --update
python scripts/prepare_corpus.py
python scripts/build_index.py
```

The index builder writes artifacts under `data/index/`, including `bm25.pkl`, `chroma/`, and `index_summary.json`.

## Uploading User Documents

Authenticated users can upload:

- PDF
- DOCX
- Markdown
- plain text

Uploads are parsed and stored under:

```text
data/uploads/user_<id>/raw/user_uploads.jsonl
data/uploads/user_<id>/processed/user_upload_chunks.jsonl
data/uploads/user_<id>/files/
data/uploads/user_<id>/index/
```

Uploads are private per user. They are not merged into the public corpus index. Query requests include them only when the client sends `include_uploaded_chunks=true`.

## Docker Deployment

The default Docker stack runs the API, client, and optionally Ollama.

### Option A: API And Client Containers With Ollama On The Host

Use this when Ollama is installed directly on the Docker host.

```powershell
Copy-Item .env.example .env
ollama serve
ollama pull llama3.2:3b
docker compose up -d --build
```

Linux/macOS:

```bash
cp .env.example .env
ollama serve
ollama pull llama3.2:3b
docker compose up -d --build
```

Open:

```text
http://localhost:8501
```

For another machine on the LAN:

```text
http://SERVER_IP:8501
```

### Option B: API, Client, And Ollama In Compose

PowerShell:

```powershell
$env:DOCKER_OLLAMA_HOST = "http://ollama:11434"
docker compose --profile ollama up -d --build
docker exec legalverifirag-ollama ollama pull llama3.2:3b
docker compose restart api
```

Linux/macOS:

```bash
DOCKER_OLLAMA_HOST=http://ollama:11434 docker compose --profile ollama up -d --build
docker exec legalverifirag-ollama ollama pull llama3.2:3b
docker compose restart api
```

Open:

```text
http://localhost:8501
```

### Option C: API Container Only

Use this when another frontend calls the API or each workstation runs Streamlit locally.

```bash
docker compose -f docker-compose.server.yml up -d --build
```

Workstation clients should set:

```env
API_BASE_URL=http://SERVER_IP:8000
```

Then run locally:

```bash
python -m streamlit run src/app.py
```

## Docker Data Persistence

The Compose stack bind-mounts:

```text
./data -> /app/data
./temp_uploads -> /app/temp_uploads
```

These files persist across container rebuilds:

- `data/legalverifirag.db`
- `data/index/chroma`
- `data/index/bm25.pkl`
- `data/index/index_summary.json`
- `data/raw`
- `data/processed`
- `data/uploads`
- `data/eval`

Back up `data/` before destructive maintenance or server migration.

## Docker Corpus Operations

After the API image is built, corpus scripts can be run inside the API container:

```bash
docker compose run --rm api python scripts/download_corpus.py --scotus --limit 10
docker compose run --rm api python scripts/prepare_corpus.py
docker compose run --rm api python scripts/build_index.py
```

Restart the API after rebuilding indices:

```bash
docker compose restart api
```

## Network Deployment

For LAN or small-firm deployment:

1. Run the full Docker stack on a server.
2. Publish port `8501` for user browser access.
3. Publish port `8000` only if direct API access is needed.
4. Keep Ollama private to the server or Docker network.
5. Put a reverse proxy such as nginx, Caddy, IIS, or a firm-approved gateway in front of the client for TLS.
6. Back up `data/` regularly.

## Health Checks And Operations

Check API health:

```bash
curl http://localhost:8000/health
```

Show Docker status:

```bash
docker compose ps
```

Follow logs:

```bash
docker compose logs -f api
docker compose logs -f client
```

Restart after `.env` changes:

```bash
docker compose up -d
docker compose restart api client
```

Stop containers while keeping data:

```bash
docker compose down
```

Rebuild after code or dependency changes:

```bash
docker compose up -d --build
```

## Weekly Corpus Updates

Windows PowerShell:

```powershell
.\scripts\weekly_corpus_update.ps1
python scripts\prepare_corpus.py
python scripts\build_index.py
```

Linux/macOS:

```bash
chmod +x scripts/weekly_corpus_update.sh
./scripts/weekly_corpus_update.sh
python scripts/prepare_corpus.py
python scripts/build_index.py
```

After rebuilding indices for a running API, restart the backend.

## Testing

Run the test suite:

```bash
python -m pytest
```

Useful targeted checks:

```bash
python -m pytest tests/test_pipeline.py
python -m pytest tests/test_api_app.py
python -m pytest tests/test_generation_context.py
python -m pytest tests/test_nli_verifier.py
```

Some live tests require a running API, Ollama, local models, or corpus/index artifacts.

## Troubleshooting

If the UI loads but queries fail, check the API:

```bash
curl http://localhost:8000/health
docker compose logs -f api
```

If Ollama is unreachable in Docker:

- Host Ollama: set `DOCKER_OLLAMA_HOST=http://host.docker.internal:11434`.
- Compose Ollama: set `DOCKER_OLLAMA_HOST=http://ollama:11434` and run with `--profile ollama`.
- LAN Ollama: set `DOCKER_OLLAMA_HOST=http://OLLAMA_SERVER_IP:11434`.

If the first verified query is slow, the NLI model may be loading. Use smaller NLI settings, prebuild the Docker image with the configured `NLI_MODEL`, or switch to `VERIFICATION_VERIFIER_MODE=heuristic` for smoke tests.

If retrieval is empty, build the corpus and index, confirm `data/index/index_summary.json` exists, and restart the API.

If CUDA memory is tight, try:

```env
NLI_BATCH_SIZE=1
NLI_MAX_LENGTH=384
NLI_UNLOAD_AFTER_REQUEST=true
OLLAMA_NUM_CTX=4096
```

## Current Limits

- Uploaded documents are separate per-user retrieval overlays, not public corpus updates.
- `src/verification/contradiction.py` is still a placeholder; contradiction handling currently comes from NLI and pipeline metadata.
- Production deployments should add TLS, monitoring, backups, stronger account administration

## License

MIT
