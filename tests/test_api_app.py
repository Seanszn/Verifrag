"""Tests for API auth, query, and history flow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import dependencies
from src.api.main import app
from src import pipeline as pipeline_module
from src.pipeline import QueryPipeline
from src.storage.database import Database


pytestmark = pytest.mark.smoke


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class _StubLLM:
    def generate_legal_answer(self, query: str, *, conversation_history=None) -> str:
        assert query == "Explain Miranda warnings"
        _ = conversation_history
        return "Miranda warnings are required before custodial interrogation."

    def generate_with_context(self, query: str, context, max_tokens=None, *, conversation_history=None) -> str:
        assert query == "Explain Miranda warnings"
        _ = context, max_tokens, conversation_history
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
        pipeline_module,
        "load_user_upload_retriever",
        lambda user_id, *, shared_embedder=None: (None, "unavailable:no_user_upload_index"),
    )
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
        assert payload["interaction"]["query"] == "Explain Miranda warnings"
        assert payload["user_message"]["role"] == "user"
        assert payload["assistant_message"]["role"] == "assistant"
        assert payload["user_message"]["interaction_id"] == payload["interaction"]["id"]
        assert payload["assistant_message"]["interaction_id"] == payload["interaction"]["id"]
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
        assert items[0]["interaction_id"] == payload["interaction"]["id"]
        assert items[1]["interaction_id"] == payload["interaction"]["id"]

        interactions = client.get(
            f"/api/conversations/{conversation_id}/interactions",
            headers=_auth_headers(token),
        )
        assert interactions.status_code == 200
        interaction_items = interactions.json()
        assert len(interaction_items) == 1
        assert interaction_items[0]["interaction"]["id"] == payload["interaction"]["id"]
        assert len(interaction_items[0]["claims"]) >= 1
        assert interaction_items[0]["claims"][0]["linked_citations"] == []
        assert interaction_items[0]["citations"] == []
        assert interaction_items[0]["claim_citation_links"] == []

    with test_db.connect() as conn:
        interaction = conn.execute(
            "SELECT conversation_id, query, response FROM interactions WHERE id = ?",
            (payload["interaction"]["id"],),
        ).fetchone()
        assert interaction is not None
        assert interaction["conversation_id"] == conversation_id
        assert interaction["query"] == "Explain Miranda warnings"
        assert interaction["response"] == "Miranda warnings are required before custodial interrogation."

        claim_count = conn.execute(
            "SELECT COUNT(*) FROM verified_claims WHERE interaction_id = ?",
            (payload["interaction"]["id"],),
        ).fetchone()[0]
        assert claim_count >= 1

        citation_count = conn.execute(
            "SELECT COUNT(*) FROM interaction_citations WHERE interaction_id = ?",
            (payload["interaction"]["id"],),
        ).fetchone()[0]
        assert citation_count == 0

        claim_link_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM claim_citation_links
            WHERE verified_claim_id IN (
                SELECT id FROM verified_claims WHERE interaction_id = ?
            )
            """,
            (payload["interaction"]["id"],),
        ).fetchone()[0]
        assert claim_link_count == 0

        conversation_state = conn.execute(
            "SELECT summary FROM conversation_state WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        assert conversation_state is not None
        assert "Explain Miranda warnings" in conversation_state["summary"]


def test_query_can_run_in_generation_only_mode(monkeypatch, tmp_path: Path):
    test_db = Database(tmp_path / "test.db")
    test_db.initialize()

    monkeypatch.setattr(dependencies, "database", test_db)
    monkeypatch.setattr(
        pipeline_module,
        "load_user_upload_retriever",
        lambda user_id, *, shared_embedder=None: (None, "unavailable:no_user_upload_index"),
    )
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
