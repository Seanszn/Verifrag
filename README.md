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
- `LLM_MODEL=llama3.1:8b`
- `OLLAMA_HOST=http://localhost:11434`
- `CHROMA_PATH=data/index/chroma`

This is already the default in `.env.example` and `src/config.py`.

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
- Optionally pull `llama3.1:8b` if `ollama` is installed

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
ollama pull llama3.1:8b
```

## Configuration

`.env.example` includes local API and UI defaults.

## Engineering Notes

- Coding standards: `CODING_STANDARDS.md`
- Cleanup priorities: `CLEANUP_PLAN.md`

## Usage (Current Repository)

```bash
# Download a small SCOTUS test batch (recommended first)
python scripts/download_corpus.py --scotus --limit 10

# Add local files to the corpus (PDF, DOCX, TXT, Markdown)
python scripts/ingest_user_files.py path/to/file.pdf path/to/notes.md --rebuild-index

# Full corpus sync examples
python scripts/download_corpus.py --all
python scripts/download_corpus.py --update

# Build ChromaDB and BM25 indices
python scripts/build_index.py

# Run backend and Streamlit client in separate terminals
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
streamlit run src/app.py
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

Use `scripts/ingest_user_files.py` to parse local PDFs, DOCX files, plain text, or Markdown into the same raw/processed JSONL corpus format used by the rest of the system.

```bash
python scripts/ingest_user_files.py ./my-brief.pdf ./draft-order.docx
python scripts/ingest_user_files.py ./uploads --recursive --rebuild-index
```

By default this writes:

- `data/raw/user_uploads.jsonl`
- `data/processed/user_upload_chunks.jsonl`

Pass `--rebuild-index` if you want the newly ingested files to be searchable immediately.

## Current Repository Status

This repository contains a mix of implemented modules and scaffolding:

- `scripts/download_corpus.py` is implemented and usable
- `scripts/build_index.py` builds local ChromaDB and BM25 indices from JSONL inputs
- `src/api/main.py` provides FastAPI auth, conversation, and query endpoints
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
