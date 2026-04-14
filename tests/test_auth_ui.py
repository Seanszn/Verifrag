"""UI tests for the backend-driven Streamlit client."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import pytest
import requests
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "src" / "app.py"


class FakeResponse:
    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self._payload = payload
        self.text = "" if payload is None else json.dumps(payload)

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self):
        if self._payload is None:
            raise ValueError("Response does not contain JSON.")
        return self._payload

    def raise_for_status(self) -> None:
        if not self.ok:
            raise requests.HTTPError(response=self)


class BackendStub:
    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], list[FakeResponse]] = {}
        self.calls: list[dict[str, object]] = []

    def queue(self, method: str, path: str, *responses: FakeResponse) -> None:
        self.routes[(method.upper(), path)] = list(responses)

    def __call__(self, method: str, url: str, headers=None, timeout=None, **kwargs):
        path = urlparse(url).path
        key = (method.upper(), path)
        self.calls.append(
            {
                "method": method.upper(),
                "path": path,
                "headers": headers or {},
                "timeout": timeout,
                "kwargs": kwargs,
            }
        )
        if key not in self.routes or not self.routes[key]:
            raise AssertionError(f"Unexpected request: {key}")

        responses = self.routes[key]
        if len(responses) > 1:
            return responses.pop(0)
        return responses[0]


@pytest.fixture
def backend_stub(monkeypatch: pytest.MonkeyPatch) -> BackendStub:
    stub = BackendStub()
    monkeypatch.setattr(requests, "request", stub)
    return stub


def test_successful_account_creation_uses_backend_register(backend_stub: BackendStub):
    backend_stub.queue(
        "POST",
        "/api/auth/register",
        FakeResponse(
            201,
            {
                "token": "new-user-token",
                "user": {"id": 1, "username": "new_lawyer", "created_at": "2026-04-14T12:00:00+00:00"},
            },
        ),
    )

    at = AppTest.from_file(str(APP_PATH)).run()
    at.button(key="header_login_Register").click().run()
    at.text_input(key="login_username").input("new_lawyer")
    at.text_input(key="login_password").input("secure_pass_123")
    at.button(key="login_submit").click().run()

    assert at.success
    assert "Registration successful!" in at.success[0].value
    assert at.session_state["show_register_notice"] is False
    assert backend_stub.calls[0]["path"] == "/api/auth/register"
    assert backend_stub.calls[0]["kwargs"]["json"] == {
        "username": "new_lawyer",
        "password": "secure_pass_123",
    }


def test_successful_login_uses_backend_auth(backend_stub: BackendStub):
    backend_stub.queue(
        "POST",
        "/api/auth/login",
        FakeResponse(
            200,
            {
                "token": "auth-token-123",
                "user": {"id": 7, "username": "valid_client", "created_at": "2026-04-14T12:00:00+00:00"},
            },
        ),
    )

    at = AppTest.from_file(str(APP_PATH)).run()
    at.text_input(key="login_username").input("valid_client")
    at.text_input(key="login_password").input("correct_password")
    at.button(key="login_submit").click().run()

    assert at.session_state["authenticated"] is True
    assert at.session_state["page"] == "home"
    assert at.session_state["user"]["username"] == "valid_client"
    assert at.session_state["auth_token"] == "auth-token-123"
    assert backend_stub.calls[0]["path"] == "/api/auth/login"
    assert backend_stub.calls[0]["kwargs"]["json"] == {
        "username": "valid_client",
        "password": "correct_password",
    }


def test_rejected_incorrect_password_uses_backend_error(backend_stub: BackendStub):
    backend_stub.queue(
        "POST",
        "/api/auth/login",
        FakeResponse(401, {"detail": "Invalid username or password."}),
    )

    at = AppTest.from_file(str(APP_PATH)).run()
    at.text_input(key="login_username").input("real_user")
    at.text_input(key="login_password").input("totally_wrong_password")
    at.button(key="login_submit").click().run()

    assert at.error
    assert "Invalid username or password" in at.error[0].value
    assert at.session_state["authenticated"] is False


def test_query_and_history_are_loaded_from_backend(backend_stub: BackendStub):
    backend_stub.queue(
        "POST",
        "/api/auth/login",
        FakeResponse(
            200,
            {
                "token": "token-xyz",
                "user": {"id": 9, "username": "case_user", "created_at": "2026-04-14T12:00:00+00:00"},
            },
        ),
    )
    backend_stub.queue(
        "GET",
        "/api/conversations",
        FakeResponse(
            200,
            [
                {
                    "id": 11,
                    "user_id": 9,
                    "title": "Existing Miranda Session",
                    "created_at": "2026-04-14T12:05:00+00:00",
                    "updated_at": "2026-04-14T12:06:00+00:00",
                }
            ],
        ),
    )
    backend_stub.queue(
        "GET",
        "/api/conversations/11/messages",
        FakeResponse(
            200,
            [
                {
                    "id": 31,
                    "conversation_id": 11,
                    "role": "user",
                    "content": "Explain Miranda warnings",
                    "created_at": "2026-04-14T12:05:01+00:00",
                    "metadata_json": None,
                },
                {
                    "id": 32,
                    "conversation_id": 11,
                    "role": "assistant",
                    "content": "The court held that Miranda warnings are required.",
                    "created_at": "2026-04-14T12:05:04+00:00",
                    "metadata_json": "{\"claim_count\":1}",
                },
            ],
        ),
    )

    at = AppTest.from_file(str(APP_PATH)).run()
    at.text_input(key="login_username").input("case_user")
    at.text_input(key="login_password").input("strong_password")
    at.button(key="login_submit").click().run()
    at.button(key="home_query").click().run()
    at.button(key="session_11").click().run()

    assert at.session_state["selected_conversation_id"] == 11
    assert len(at.session_state["conversation_messages"]) == 2
    assert at.session_state["conversation_messages"][1]["content"] == (
        "The court held that Miranda warnings are required."
    )
    assert [call["path"] for call in backend_stub.calls] == [
        "/api/auth/login",
        "/api/conversations",
        "/api/conversations/11/messages",
    ]


def test_submit_query_uses_backend_response_instead_of_placeholder(backend_stub: BackendStub):
    backend_stub.queue(
        "POST",
        "/api/auth/login",
        FakeResponse(
            200,
            {
                "token": "token-query",
                "user": {"id": 4, "username": "query_user", "created_at": "2026-04-14T12:00:00+00:00"},
            },
        ),
    )
    backend_stub.queue("GET", "/api/conversations", FakeResponse(200, []))
    backend_stub.queue(
        "POST",
        "/api/query",
        FakeResponse(
            200,
            {
                "conversation": {
                    "id": 21,
                    "user_id": 4,
                    "title": "Explain Miranda warnings",
                    "created_at": "2026-04-14T12:10:00+00:00",
                    "updated_at": "2026-04-14T12:10:02+00:00",
                },
                "user_message": {
                    "id": 41,
                    "conversation_id": 21,
                    "role": "user",
                    "content": "Explain Miranda warnings",
                    "created_at": "2026-04-14T12:10:01+00:00",
                    "metadata_json": None,
                },
                "assistant_message": {
                    "id": 42,
                    "conversation_id": 21,
                    "role": "assistant",
                    "content": "The court held that Miranda warnings are required.",
                    "created_at": "2026-04-14T12:10:02+00:00",
                    "metadata_json": "{\"claim_count\":1}",
                },
                "pipeline": {
                    "llm_backend_status": "ok",
                    "retrieval_backend_status": "unavailable:no_indices",
                    "verification_backend_status": "skipped:no_retriever",
                    "claim_count": 1,
                },
            },
        ),
    )

    at = AppTest.from_file(str(APP_PATH)).run()
    at.text_input(key="login_username").input("query_user")
    at.text_input(key="login_password").input("strong_password")
    at.button(key="login_submit").click().run()
    at.button(key="home_query").click().run()
    at.text_area(key="current_query").input("Explain Miranda warnings")
    at.button(key="submit_query").click().run()

    assert at.session_state["selected_conversation_id"] == 21
    assert len(at.session_state["conversation_messages"]) == 2
    assert at.session_state["conversation_messages"][1]["content"] == (
        "The court held that Miranda warnings are required."
    )
    assert at.session_state["last_pipeline"]["claim_count"] == 1
    assert at.session_state["current_query"] == ""
    assert at.session_state["page"] == "response"
    assert "Placeholder response" not in at.session_state["conversation_messages"][1]["content"]


def test_submit_query_appends_to_active_conversation_without_history_reload(backend_stub: BackendStub):
    backend_stub.queue(
        "POST",
        "/api/auth/login",
        FakeResponse(
            200,
            {
                "token": "token-followup",
                "user": {"id": 12, "username": "followup_user", "created_at": "2026-04-14T12:00:00+00:00"},
            },
        ),
    )
    backend_stub.queue(
        "GET",
        "/api/conversations",
        FakeResponse(
            200,
            [
                {
                    "id": 33,
                    "user_id": 12,
                    "title": "Miranda follow-up",
                    "created_at": "2026-04-14T12:00:00+00:00",
                    "updated_at": "2026-04-14T12:05:00+00:00",
                }
            ],
        ),
    )
    backend_stub.queue(
        "GET",
        "/api/conversations/33/messages",
        FakeResponse(
            200,
            [
                {
                    "id": 51,
                    "conversation_id": 33,
                    "role": "user",
                    "content": "Explain Miranda warnings.",
                    "created_at": "2026-04-14T12:00:01+00:00",
                    "metadata_json": None,
                },
                {
                    "id": 52,
                    "conversation_id": 33,
                    "role": "assistant",
                    "content": "Miranda requires warnings during custodial interrogation.",
                    "created_at": "2026-04-14T12:00:03+00:00",
                    "metadata_json": "{\"claim_count\":1}",
                },
            ],
        ),
    )
    backend_stub.queue(
        "POST",
        "/api/query",
        FakeResponse(
            200,
            {
                "conversation": {
                    "id": 33,
                    "user_id": 12,
                    "title": "Miranda follow-up",
                    "created_at": "2026-04-14T12:00:00+00:00",
                    "updated_at": "2026-04-14T12:06:30+00:00",
                },
                "user_message": {
                    "id": 53,
                    "conversation_id": 33,
                    "role": "user",
                    "content": "What about the public safety exception?",
                    "created_at": "2026-04-14T12:06:00+00:00",
                    "metadata_json": None,
                },
                "assistant_message": {
                    "id": 54,
                    "conversation_id": 33,
                    "role": "assistant",
                    "content": "The public safety exception can permit limited unwarned questioning.",
                    "created_at": "2026-04-14T12:06:30+00:00",
                    "metadata_json": "{\"claim_count\":1}",
                },
                "pipeline": {
                    "llm_backend_status": "ok",
                    "retrieval_backend_status": "unavailable:no_indices",
                    "verification_backend_status": "skipped:no_retriever",
                    "claim_count": 1,
                    "conversation_context_message_count": 2,
                },
            },
        ),
    )

    at = AppTest.from_file(str(APP_PATH)).run()
    at.text_input(key="login_username").input("followup_user")
    at.text_input(key="login_password").input("strong_password")
    at.button(key="login_submit").click().run()
    at.button(key="home_query").click().run()
    at.button(key="session_33").click().run()
    at.text_area(key="current_query").input("What about the public safety exception?")
    at.button(key="submit_query").click().run()

    assert at.session_state["selected_conversation_id"] == 33
    assert len(at.session_state["conversation_messages"]) == 4
    assert at.session_state["conversation_messages"][3]["content"] == (
        "The public safety exception can permit limited unwarned questioning."
    )
    assert at.session_state["last_pipeline"]["conversation_context_message_count"] == 2
    assert [call["path"] for call in backend_stub.calls] == [
        "/api/auth/login",
        "/api/conversations",
        "/api/conversations/33/messages",
        "/api/query",
    ]
