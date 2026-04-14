"""Tests for API auth, query, and history flow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import dependencies
from src.api.main import app
from src.pipeline import QueryPipeline
from src.storage.database import Database


pytestmark = pytest.mark.smoke


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class _StubLLM:
    def generate_legal_answer(self, query: str) -> str:
        assert query == "Explain Miranda warnings"
        return "Miranda warnings are required before custodial interrogation."

    def generate_with_context(self, query: str, context, max_tokens=None) -> str:
        assert query == "Explain Miranda warnings"
        _ = context, max_tokens
        return "Miranda warnings are required before custodial interrogation."


class _EmptyRetriever:
    def retrieve(self, query: str, k: int = 10):
        assert query == "Explain Miranda warnings"
        _ = k
        return []


def test_register_query_and_history(monkeypatch, tmp_path: Path):
    test_db = Database(tmp_path / "test.db")
    test_db.initialize()

    monkeypatch.setattr(dependencies, "database", test_db)
    monkeypatch.setattr(
        dependencies,
        "pipeline",
        QueryPipeline(
            db=test_db,
            llm=_StubLLM(),
            retriever=_EmptyRetriever(),
        ),
    )

    with TestClient(app) as client:
        register = client.post(
            "/api/auth/register",
            json={"username": "alice", "password": "password123"},
        )
        assert register.status_code == 201
        token = register.json()["token"]

        query = client.post(
            "/api/query",
            headers=_auth_headers(token),
            json={"query": "Explain Miranda warnings"},
        )
        assert query.status_code == 200
        payload = query.json()
        assert payload["conversation"]["title"] == "Explain Miranda warnings"
        assert payload["user_message"]["role"] == "user"
        assert payload["assistant_message"]["role"] == "assistant"
        assert payload["pipeline"]["claim_count"] >= 1
        assert json.loads(payload["assistant_message"]["metadata_json"])["claim_count"] >= 1

        conversation_id = payload["conversation"]["id"]
        conversations = client.get(
            "/api/conversations",
            headers=_auth_headers(token),
        )
        assert conversations.status_code == 200
        assert len(conversations.json()) == 1

        messages = client.get(
            f"/api/conversations/{conversation_id}/messages",
            headers=_auth_headers(token),
        )
        assert messages.status_code == 200
        items = messages.json()
        assert len(items) == 2
        assert items[0]["role"] == "user"
        assert items[1]["role"] == "assistant"


def test_query_can_run_in_generation_only_mode(monkeypatch, tmp_path: Path):
    test_db = Database(tmp_path / "test.db")
    test_db.initialize()

    monkeypatch.setattr(dependencies, "database", test_db)
    monkeypatch.setattr(
        dependencies,
        "pipeline",
        QueryPipeline(
            db=test_db,
            llm=_StubLLM(),
            retriever=_EmptyRetriever(),
            enable_verification=False,
        ),
    )

    with TestClient(app) as client:
        register = client.post(
            "/api/auth/register",
            json={"username": "bob", "password": "password123"},
        )
        assert register.status_code == 201
        token = register.json()["token"]

        query = client.post(
            "/api/query",
            headers=_auth_headers(token),
            json={"query": "Explain Miranda warnings"},
        )
        assert query.status_code == 200
        payload = query.json()
        assert payload["pipeline"]["verification_enabled"] is False
        assert payload["pipeline"]["verification_backend_status"] == "disabled:config"
        assert payload["pipeline"]["claim_count"] == 0
        assert payload["pipeline"]["claims"] == []
