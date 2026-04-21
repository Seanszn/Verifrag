from __future__ import annotations

import os
from uuid import uuid4

import pytest
import requests


pytestmark = pytest.mark.integration


def _live_base_url() -> str | None:
    return os.getenv("LIVE_API_BASE_URL")


@pytest.mark.skipif(not _live_base_url(), reason="LIVE_API_BASE_URL is not set.")
def test_live_http_query_round_trip():
    base_url = _live_base_url()
    assert base_url is not None

    username = os.getenv("LIVE_API_USERNAME", f"live_probe_{uuid4().hex[:8]}")
    password = os.getenv("LIVE_API_PASSWORD", "live_probe_password_123")
    query = os.getenv("LIVE_API_QUERY", "test")
    connect_timeout = int(os.getenv("LIVE_API_CONNECT_TIMEOUT_SECONDS", "10"))
    request_timeout = int(os.getenv("LIVE_API_REQUEST_TIMEOUT_SECONDS", "30"))
    query_timeout = int(os.getenv("LIVE_API_QUERY_TIMEOUT_SECONDS", "180"))
    request_id = uuid4().hex

    def request(method: str, path: str, *, token: str | None = None, timeout=(10, 30), **kwargs):
        headers = {"X-Request-ID": request_id}
        if "json" in kwargs:
            headers["Content-Type"] = "application/json"
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return requests.request(
            method,
            f"{base_url.rstrip('/')}{path}",
            headers=headers,
            timeout=timeout,
            **kwargs,
        )

    auth_response = request(
        "POST",
        "/api/auth/register",
        timeout=(connect_timeout, request_timeout),
        json={"username": username, "password": password},
    )
    assert auth_response.status_code in {200, 201, 409}

    if auth_response.status_code == 409:
        auth_response = request(
            "POST",
            "/api/auth/login",
            timeout=(connect_timeout, request_timeout),
            json={"username": username, "password": password},
        )

    assert auth_response.ok, auth_response.text
    token = auth_response.json()["token"]

    conversations_response = request(
        "GET",
        "/api/conversations",
        token=token,
        timeout=(connect_timeout, request_timeout),
    )
    assert conversations_response.ok, conversations_response.text

    query_response = request(
        "POST",
        "/api/query",
        token=token,
        timeout=(connect_timeout, query_timeout),
        json={"query": query, "conversation_id": None},
    )
    assert query_response.ok, query_response.text
    payload = query_response.json()
    pipeline = payload["pipeline"]

    assert payload["interaction"]["query"] == query
    assert pipeline["request_id"] == request_id
    assert "timings_ms" in pipeline
