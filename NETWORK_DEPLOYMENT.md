# LegalVerifiRAG Network Deployment

This guide explains the supported ways to run LegalVerifiRAG for multiple users on a LAN or private firm network.

## Deployment Options

| Option | Use When | What Runs On The Server | What Users Open |
| --- | --- | --- | --- |
| Full Docker stack | You want the simplest repeatable deployment | FastAPI API, Streamlit client, optional Ollama | `http://SERVER_IP:8501` |
| Docker API only | Users run the Streamlit client on their own machines | FastAPI API | Local Streamlit configured with `API_BASE_URL=http://SERVER_IP:8000` |
| Native local run | Development or a single workstation | FastAPI API and Streamlit directly in Python | `http://127.0.0.1:8501` |

For a shared network deployment, use the full Docker stack unless there is a specific reason to run the Streamlit client on each workstation.

## Network Architecture

```text
User browser
  |
  | http://SERVER_IP:8501
  v
Streamlit client container
  |
  | http://api:8000 inside Docker network
  v
FastAPI backend container
  |
  | SQLite database, Chroma index, uploads in ./data and ./temp_uploads
  v
Mounted server directories

FastAPI also calls Ollama at one of:
- http://host.docker.internal:11434 when Ollama runs on the server host
- http://ollama:11434 when Ollama runs as the optional Compose service
- http://OLLAMA_SERVER_IP:11434 when Ollama runs on another LAN machine
```

The client should never connect directly to SQLite, Chroma, or Ollama. Only the API owns authentication, conversations, retrieval, generation, verification, and uploads.

## Prerequisites

- Docker Engine and Docker Compose v2 on the server.
- A reachable Ollama instance with the configured model pulled.
- Inbound firewall access to the chosen published ports:
  - `8501/tcp` for the browser UI.
  - `8000/tcp` only if clients or test tools need direct API access.
  - `11434/tcp` only if other machines need direct Ollama access.
- Existing retrieval index files under `data/index` if you want retrieval-backed answers immediately.

## Prepare Configuration

Run commands from the `legalverifirag/` directory.

Create `.env` if it does not exist:

```powershell
Copy-Item .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

Important values:

```env
LLM_MODEL=llama3.2:3b
OLLAMA_HOST=http://localhost:11434
DOCKER_OLLAMA_HOST=http://host.docker.internal:11434
API_BASE_URL=http://127.0.0.1:8000
CLIENT_API_BASE_URL=http://api:8000
API_PUBLISHED_PORT=8000
CLIENT_PUBLISHED_PORT=8501
DATABASE_PATH=data/legalverifirag.db
CHROMA_PATH=data/index/chroma
ENABLE_VERIFICATION=true
COURTLISTENER_TOKEN=
```

Notes:

- `OLLAMA_HOST` is used by native local runs.
- `DOCKER_OLLAMA_HOST` is mapped into the API container as `OLLAMA_HOST`.
- `CLIENT_API_BASE_URL=http://api:8000` is mapped into the Dockerized Streamlit client as `API_BASE_URL`.
- Docker Compose overrides `DATABASE_PATH` and `CHROMA_PATH` inside the API container to `/app/data/...` while bind-mounting the server's `./data` directory.
- If users run Streamlit outside Docker, set their client-side `API_BASE_URL` to `http://SERVER_IP:8000`.
- Keep `COURTLISTENER_TOKEN` blank unless you are downloading or updating the corpus.

## Option A: Full Docker Stack With Ollama On The Host

Use this when Ollama is installed directly on the same server that runs Docker.

1. Start Ollama on the server host:

   ```powershell
   ollama serve
   ollama pull llama3.2:3b
   ```

2. Set the API to reach host Ollama:

   ```env
   DOCKER_OLLAMA_HOST=http://host.docker.internal:11434
   ```

3. Build and start the API and browser client:

   ```powershell
   docker compose up -d --build
   ```

4. Verify:

   ```powershell
   docker compose ps
   curl http://localhost:8000/health
   ```

5. From another workstation on the same network, open:

   ```text
   http://SERVER_IP:8501
   ```

## Option B: Full Docker Stack With Ollama In Docker

Use this when the server can run Ollama as a container.

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

Then open:

```text
http://SERVER_IP:8501
```

The Ollama model files persist in the named Docker volume `legalverifirag_ollama`.

## Option C: API Container Only

Use this when each workstation runs the Streamlit client locally, or when another frontend will call the API.

```powershell
docker compose -f docker-compose.server.yml up -d --build
```

Workstations then need:

```env
API_BASE_URL=http://SERVER_IP:8000
```

Start the local client on a workstation:

```powershell
python -m streamlit run src/app.py
```

## Option D: Native Development Run

Use this for development or a single machine.

PowerShell:

```powershell
.\scripts\setup_local.ps1
.\scripts\run_api.ps1
.\scripts\run_local.ps1
```

Linux/macOS:

```bash
chmod +x scripts/setup_local.sh scripts/run_api.sh scripts/run_local.sh
./scripts/setup_local.sh
./scripts/run_api.sh
./scripts/run_local.sh
```

## Data And Persistence

The Docker stack bind-mounts:

- `./data` to `/app/data`
- `./temp_uploads` to `/app/temp_uploads`

This keeps these files on the server even when containers are rebuilt:

- SQLite database: `data/legalverifirag.db`
- Chroma vector store: `data/index/chroma`
- processed corpus and BM25 index files under `data`
- uploaded user documents under `data/uploads`
- temporary upload staging under `temp_uploads`

Back up `data/` before server migrations or destructive maintenance.

## Building Or Updating The Retrieval Index

If `data/index` is empty, the API can still call the LLM, but retrieval-backed verification will be limited.

For a small test corpus:

```powershell
python scripts/download_corpus.py --scotus --limit 10
python scripts/build_index.py
```

For Docker-based operations, run the same scripts in the API container after it is built:

```powershell
docker compose run --rm api python scripts/download_corpus.py --scotus --limit 10
docker compose run --rm api python scripts/build_index.py
```

## Firewall And Browser Access

On the server firewall, allow inbound TCP for the ports you publish:

- `8501` for the Streamlit web UI.
- `8000` for direct API access, health checks, or local clients.

For production or firm-wide use, put a reverse proxy in front of the services:

- publish `https://legalverifirag.company.local` to the Streamlit client on `8501`
- optionally publish `https://legalverifirag-api.company.local` to the API on `8000`
- terminate TLS at nginx, Caddy, IIS, or another approved gateway

Do not expose Ollama directly to untrusted networks.

## Common Operations

Show status:

```powershell
docker compose ps
```

Follow API logs:

```powershell
docker compose logs -f api
```

Follow client logs:

```powershell
docker compose logs -f client
```

Restart after `.env` changes:

```powershell
docker compose up -d
docker compose restart api client
```

Stop containers but keep data:

```powershell
docker compose down
```

Rebuild after code changes:

```powershell
docker compose up -d --build
```

## Troubleshooting

If the UI loads but queries fail, check API reachability:

```powershell
curl http://SERVER_IP:8000/health
docker compose logs -f api
```

If API logs show Ollama connection errors:

- Host Ollama: use `DOCKER_OLLAMA_HOST=http://host.docker.internal:11434`.
- Docker Ollama: use `DOCKER_OLLAMA_HOST=http://ollama:11434` and run with `--profile ollama`.
- LAN Ollama: use `DOCKER_OLLAMA_HOST=http://OLLAMA_SERVER_IP:11434` and confirm the server firewall permits access.

If the first query is slow, the container may be downloading Hugging Face models or warming the local model. Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_LOCAL_FILES_ONLY=1` only after the required models are already cached.

If retrieval results are empty, confirm `data/index/chroma` and the BM25 index files exist, then rebuild the index if needed.
