from __future__ import annotations

import asyncio
import logging

from fastapi.testclient import TestClient
from src.api import main as api_main


class _FakeLLM:
    def diagnostics(self) -> dict[str, object]:
        return {
            "provider": "ollama",
            "host": "http://localhost:11434",
            "model": "llama3.1:8b",
            "reachable": True,
            "model_available": True,
        }


class _FakePipeline:
    llm = _FakeLLM()
    enable_verification = True
    retriever_status = "ok"

    def preload_models(self) -> dict[str, str]:
        return {
            "retrieval_embedder": "ok",
            "verification_model": "ok",
        }


def test_healthcheck_reports_llm_diagnostics(monkeypatch):
    monkeypatch.setattr(api_main.dependencies, "pipeline", _FakePipeline())

    payload = asyncio.run(api_main.healthcheck())

    assert payload == {
        "status": "ok",
        "llm": {
            "provider": "ollama",
            "host": "http://localhost:11434",
            "model": "llama3.1:8b",
            "reachable": True,
            "model_available": True,
        },
    }


def test_health_request_sets_response_request_id(monkeypatch, caplog):
    monkeypatch.setattr(api_main.dependencies, "pipeline", _FakePipeline())

    with caplog.at_level(logging.INFO, logger="src.api.main"):
        with TestClient(api_main.app) as client:
            response = client.get("/health", headers={"X-Request-ID": "health-test-id"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "health-test-id"
    assert float(response.headers["X-Process-Time-MS"]) >= 0.0
    assert "api.request_start request_id=health-test-id" in caplog.text
    assert "api.request_complete request_id=health-test-id" in caplog.text
