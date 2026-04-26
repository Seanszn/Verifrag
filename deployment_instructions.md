# Deployment Instructions

The deployment paths have been consolidated in [NETWORK_DEPLOYMENT.md](NETWORK_DEPLOYMENT.md).

Use that guide for:

- full network deployment with Docker Compose
- API-only Docker deployment
- optional Ollama-in-Docker deployment
- native local development runs
- firewall, persistence, backup, and troubleshooting notes

Quick start for a shared network server:

```powershell
Copy-Item .env.example .env
docker compose up -d --build
curl http://localhost:8000/health
```

Then open the Streamlit client from a workstation:

```text
http://SERVER_IP:8501
```
