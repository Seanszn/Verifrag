# LegalVerifiRAG

Claim-level verification system for legal AI. Detects hallucinations in AI-generated legal analysis by decomposing responses into atomic claims and verifying each against authoritative legal sources.

## Overview

Commercial legal AI tools hallucinate at 17-33% rates (Stanford Law, 2024). They cite documents but do not verify that claims about those documents are accurate. LegalVerifiRAG adds a verification layer that:

1. Retrieves relevant legal passages
2. Generates a response grounded in those passages
3. Decomposes the response into typed legal claims
4. Verifies each claim against the retrieved sources using NLI
5. Outputs the response with inline verification badges

## Local-First Default

Local deployment is the default configuration:

- `DEPLOYMENT_MODE=local`
- `LLM_MODEL=llama3.2:3b`
- `OLLAMA_HOST=http://localhost:11434`
- `LLM_MAX_TOKENS=1024`
- `LLM_REQUEST_TIMEOUT_SECONDS=150`
- `OLLAMA_NUM_CTX=4096`
- `API_QUERY_TIMEOUT_SECONDS=180`
- `APP_LOG_LEVEL=INFO`
- `CHROMA_PATH=data/index/chroma`
- `ENABLE_VERIFICATION=true` for the full claim-analysis pipeline

This is already the default in `.env.example`, the local setup/run scripts, and `src/config.py`.

## Quick Start (Local Default)

Run from the `legalverifirag/` directory.

### Windows (PowerShell)

```powershell
.\scripts\setup_local.ps1
.\scripts\run_api.ps1
.\scripts\run_local.ps1
```

### Linux (Bash)

```bash
chmod +x scripts/setup_local.sh scripts/run_api.sh scripts/run_local.sh
./scripts/setup_local.sh
./scripts/run_api.sh
./scripts/run_local.sh
```

## What The Setup Scripts Do

- Create a virtual environment (`venv`) if missing
- Install Python dependencies from `requirements.txt`
- Download the spaCy model (`en_core_web_sm`)
- Create `.env` from `.env.example` if missing
- Ensure `.env` defaults to local mode
- Create local data directories (`data/raw`, `data/processed`, `data/index`, `data/eval`)
- Optionally pull `llama3.2:3b` if `ollama` is installed

## Manual Setup (Equivalent)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows PowerShell: .\venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Create local config
cp .env.example .env   # Windows PowerShell: Copy-Item .env.example .env

# Install Ollama separately (https://ollama.com), then pull local model
ollama pull llama3.2:3b
```

## Configuration

`.env.example` includes local API and UI defaults.

Local setup now enables claim verification by default. Set `ENABLE_VERIFICATION=false` when you want faster generation-only smoke tests.
The default query path now waits longer than standard API calls: regular API requests default to 30 seconds, `/api/query` defaults to 180 seconds, and Ollama generation defaults to 150 seconds.
If local generation is still too slow, reduce `LLM_MAX_TOKENS` further or disable verification for smoke tests.
Backend request logging is now correlated by `X-Request-ID`, includes per-request duration headers, and emits query stage logs for retrieval, generation, claim decomposition, and verification.

## Engineering Notes

- Coding standards: `CODING_STANDARDS.md`
- Cleanup priorities: `CLEANUP_PLAN.md`

## Usage (Current Repository)

```bash
# Download a small SCOTUS test batch (recommended first)
python scripts/download_corpus.py --scotus --limit 10

# Full corpus sync examples
python scripts/download_corpus.py --all
python scripts/download_corpus.py --update

# Build ChromaDB and BM25 indices
python scripts/build_index.py

# Run backend and Streamlit client in separate terminals
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
streamlit run src/app.py
```

In the local startup scripts, `/api/query` runs with claim decomposition and NLI verification enabled by default.

## Network Deployment

For shared LAN or server deployment, use the consolidated guide:

- [NETWORK_DEPLOYMENT.md](NETWORK_DEPLOYMENT.md)

The default Docker stack runs:

- FastAPI backend on container port `8000`
- Streamlit client on container port `8501`
- optional Ollama service with `docker compose --profile ollama`

Basic server start:

```bash
cp .env.example .env
docker compose up -d --build
```

## Weekly Corpus Updates (Incremental)

Use the wrapper scripts below to run `scripts/download_corpus.py --update` on a schedule. They use the existing sync state to fetch only new/updated cases and write timestamped logs.

### Windows (PowerShell / Task Scheduler)

```powershell
.\scripts\weekly_corpus_update.ps1
```

Optional flags:

```powershell
.\scripts\weekly_corpus_update.ps1 -VerboseDownload
.\scripts\weekly_corpus_update.ps1 -Limit 100
.\scripts\weekly_corpus_update.ps1 -OutputDir data\raw
```

Task Scheduler action example:

```text
Program/script: powershell.exe
Add arguments: -ExecutionPolicy Bypass -File "E:\path\to\legalverifirag\scripts\weekly_corpus_update.ps1"
Start in: E:\path\to\legalverifirag
```

### Linux (Bash / cron)

```bash
chmod +x scripts/weekly_corpus_update.sh
./scripts/weekly_corpus_update.sh
```

Optional env vars:

```bash
VERBOSE_DOWNLOAD=1 ./scripts/weekly_corpus_update.sh
LIMIT=100 ./scripts/weekly_corpus_update.sh
OUTPUT_DIR=data/raw ./scripts/weekly_corpus_update.sh
```

Cron example (weekly Monday at 2:00 AM):

```cron
0 2 * * 1 cd /path/to/legalverifirag && ./scripts/weekly_corpus_update.sh
```

Note: This updates the corpus JSONL files. Rebuild search indices after updates with `scripts/build_index.py`.

## Add Your Own Files

Upload local PDFs, DOCX files, plain text, or Markdown through the Streamlit client or by posting to `POST /api/uploads`. The backend parses the files and stores them in a per-user workspace.

```bash
# Start the backend
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# Then use the Upload Documents page in the Streamlit client
streamlit run src/app.py
```

By default uploaded files are written under:

- `data/uploads/user_<id>/raw/user_uploads.jsonl`
- `data/uploads/user_<id>/processed/user_upload_chunks.jsonl`
- `data/uploads/user_<id>/files/`

The current branch ingests uploads server-side and stores chunked JSONL output, but it does not yet rebuild the main retrieval index automatically.

## Current Repository Status

This repository contains a mix of implemented modules and scaffolding:

- `scripts/download_corpus.py` is implemented and usable
- `scripts/build_index.py` builds local ChromaDB and BM25 indices from JSONL inputs
- `src/api/main.py` provides FastAPI auth, conversation, query, and upload endpoints
- `src/app.py` is a thin Streamlit client for the backend API
- `src/pipeline.py` runs server-side query orchestration and persists history

The setup and run scripts are still useful for provisioning a new machine and standardizing local configuration.

## Project Structure

```text
legalverifirag/
|-- src/
|   |-- api/            # FastAPI backend
|   |-- auth/           # Password hashing
|   |-- client/         # Streamlit client helpers and UI
|   |-- storage/        # SQLite persistence
|   |-- ingestion/      # Document downloading and processing
|   |-- indexing/       # Vector stores and search indices
|   |-- retrieval/      # Hybrid search
|   |-- generation/     # LLM backends and prompts
|   |-- verification/   # Claim verification algorithms
|   |-- pipeline.py     # Server-side orchestration
|   `-- app.py          # Streamlit client UI
|-- data/
|   |-- raw/            # Downloaded cases and statutes
|   |-- processed/      # Chunked documents
|   |-- index/          # ChromaDB and BM25 indices
|   `-- eval/           # Test datasets
|-- tests/              # Unit tests
`-- scripts/            # Utility scripts
```

## License

MIT
