# Backend Deployment Instructions

This project already includes Docker artifacts for the backend:

- `Dockerfile.server`
- `docker-compose.server.yml`

The easiest deployment path is:

1. Run the FastAPI backend in Docker on the central server.
2. Run Ollama on the same server host, or point the backend at another Ollama host.
3. Point each client app at the backend with `API_BASE_URL`.

## What The Backend Does

The backend server exposes:

- `POST /api/auth/register`
- `POST /api/auth/login`
- `POST /api/auth/logout`
- `GET /api/conversations`
- `GET /api/conversations/{conversation_id}/messages`
- `POST /api/query`
- `GET /health`

The backend owns:

- authentication
- conversation history
- query execution
- retrieval / generation / verification pipeline execution

The client should not connect directly to the database or Ollama.

## Quick Docker Deployment

Run these steps from the `legalverifirag/` directory.

### 1. Create `.env`

If `.env` does not exist yet:

```powershell
Copy-Item .env.example .env
```

or on Linux:

```bash
cp .env.example .env
```

At minimum, check these values in `.env`:

```env
DEPLOYMENT_MODE=local
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
OLLAMA_HOST=http://host.docker.internal:11434
API_HOST=0.0.0.0
API_PORT=8000
AUTH_TOKEN_TTL_HOURS=24
DATABASE_PATH=/app/data/legalverifirag.db
CHROMA_PATH=/app/data/index/chroma
CHROMA_COLLECTION=legal_chunks
CHROMA_DISTANCE=cosine
CHROMA_BATCH_SIZE=100
```

Notes:

- `docker-compose.server.yml` already overrides `API_HOST`, `API_PORT`, `DATABASE_PATH`, `CHROMA_PATH`, and `OLLAMA_HOST` for the container.
- The compose file expects Ollama to be reachable from the container at `http://host.docker.internal:11434`.
- If Ollama is running on another machine, change `OLLAMA_HOST` to that machine's reachable URL.

### 2. Start Ollama

The current backend is configured for Ollama-only deployment.

You need Ollama running somewhere reachable by the backend, with the configured model pulled.

Example on the backend host:

```powershell
ollama serve
ollama pull llama3.1:8b
```

If you prefer the repo default in `src/config.py`, substitute `deepseek-r1:8b` and set `.env` to match.

### 3. Build And Start The Backend Container

```powershell
docker compose -f docker-compose.server.yml up -d --build
```

This starts the FastAPI backend on port `8000`.

The compose file mounts:

- `./data` into `/app/data`
- `./temp_uploads` into `/app/temp_uploads`

So database and index files persist outside the container.

### 4. Verify The Backend

Check the container:

```powershell
docker compose -f docker-compose.server.yml ps
```

Check logs:

```powershell
docker compose -f docker-compose.server.yml logs -f api
```

Check health:

```powershell
curl http://SERVER_IP:8000/health
```

Expected response:

```json
{"status":"ok"}
```

### 5. Stop The Backend

```powershell
docker compose -f docker-compose.server.yml down
```

## Non-Docker Backend Run

If you want to run the backend directly on the server host instead of Docker:

### 1. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Set environment variables

Minimum useful values:

```powershell
$env:API_HOST = "0.0.0.0"
$env:API_PORT = "8000"
$env:OLLAMA_HOST = "http://127.0.0.1:11434"
```

### 3. Start the API

```powershell
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## What The Client App Needs To Do

The client app only needs to know the backend URL.

Set this on each client machine:

```env
API_BASE_URL=http://SERVER_IP:8000
```

Then run the client app:

```powershell
python -m streamlit run src/app.py
```

The client does not need:

- a local database
- a local Ollama instance
- a local vector store

The current client flow now uses the backend for:

- login / registration
- logout
- conversation list
- message history
- query submission

## Network Requirements

The client machine must be able to reach:

- `http://SERVER_IP:8000`

For a business network deployment, make sure:

- inbound TCP `8000` is open on the backend host, or
- the backend is placed behind a reverse proxy and published on a different port / hostname

Recommended production improvement:

- put the API behind `nginx`, IIS, or another reverse proxy
- terminate TLS there
- expose an internal DNS name such as `http://legalverifirag-api.company.local` or preferably `https://...`

## Current Limitations

- The backend Docker setup only covers the FastAPI server. It does not start Ollama for you.
- Retrieval quality depends on index files already existing under `data/index`.
- If no indices are available, the backend can still answer through the LLM, but retrieval-backed verification will be limited.
- The current client upload page is not wired to the backend yet. Auth, query, and history are wired; upload integration is the next step.

## Recommended Minimal Enterprise Setup

For the simplest central-server deployment:

1. Install Docker on the backend server.
2. Install Ollama on the backend host.
3. Pull the model you want to use.
4. Start the backend with:

```powershell
docker compose -f docker-compose.server.yml up -d --build
```

5. Verify:

```powershell
curl http://SERVER_IP:8000/health
```

6. On each client machine, set:

```env
API_BASE_URL=http://SERVER_IP:8000
```

7. Run the Streamlit client:

```powershell
python -m streamlit run src/app.py
```
