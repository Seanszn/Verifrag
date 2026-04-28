from __future__ import annotations

import os
from io import BytesIO
from uuid import uuid4

import pytest
import requests


pytestmark = pytest.mark.integration


def _live_base_url() -> str | None:
    return os.getenv("LIVE_API_BASE_URL")


def _live_timeouts() -> tuple[int, int, int]:
    return (
        int(os.getenv("LIVE_API_CONNECT_TIMEOUT_SECONDS", "10")),
        int(os.getenv("LIVE_API_REQUEST_TIMEOUT_SECONDS", "30")),
        int(os.getenv("LIVE_API_QUERY_TIMEOUT_SECONDS", "180")),
    )


def _live_request(base_url: str, request_id: str, method: str, path: str, *, token: str | None = None, timeout=(10, 30), **kwargs):
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


def _register_or_login_live_user(
    *,
    base_url: str,
    request_id: str,
    username: str,
    password: str,
    connect_timeout: int,
    request_timeout: int,
) -> str:
    auth_response = _live_request(
        base_url,
        request_id,
        "POST",
        "/api/auth/register",
        timeout=(connect_timeout, request_timeout),
        json={"username": username, "password": password},
    )
    assert auth_response.status_code in {200, 201, 409}

    if auth_response.status_code == 409:
        auth_response = _live_request(
            base_url,
            request_id,
            "POST",
            "/api/auth/login",
            timeout=(connect_timeout, request_timeout),
            json={"username": username, "password": password},
        )

    assert auth_response.ok, auth_response.text
    return auth_response.json()["token"]


@pytest.mark.skipif(not _live_base_url(), reason="LIVE_API_BASE_URL is not set.")
def test_live_http_query_round_trip():
    base_url = _live_base_url()
    assert base_url is not None

    username = os.getenv("LIVE_API_USERNAME", f"live_probe_{uuid4().hex[:8]}")
    password = os.getenv("LIVE_API_PASSWORD", "live_probe_password_123")
    query = os.getenv("LIVE_API_QUERY", "test")
    connect_timeout, request_timeout, query_timeout = _live_timeouts()
    request_id = uuid4().hex

    token = _register_or_login_live_user(
        base_url=base_url,
        request_id=request_id,
        username=username,
        password=password,
        connect_timeout=connect_timeout,
        request_timeout=request_timeout,
    )

    conversations_response = _live_request(
        base_url,
        request_id,
        "GET",
        "/api/conversations",
        token=token,
        timeout=(connect_timeout, request_timeout),
    )
    assert conversations_response.ok, conversations_response.text

    query_response = _live_request(
        base_url,
        request_id,
        "POST",
        "/api/query",
        token=token,
        timeout=(connect_timeout, query_timeout),
        json={
            "query": query,
            "conversation_id": None,
            "include_uploaded_chunks": True,
        },
    )
    assert query_response.ok, query_response.text
    payload = query_response.json()
    pipeline = payload["pipeline"]

    assert payload["interaction"]["query"] == query
    assert pipeline["request_id"] == request_id
    assert "timings_ms" in pipeline


@pytest.mark.skipif(not _live_base_url(), reason="LIVE_API_BASE_URL is not set.")
def test_live_http_user_upload_query_is_grounded_and_verified():
    base_url = _live_base_url()
    assert base_url is not None

    username = f"live_upload_probe_{uuid4().hex[:8]}"
    password = "live_probe_password_123"
    connect_timeout, request_timeout, query_timeout = _live_timeouts()
    request_id = uuid4().hex
    token = _register_or_login_live_user(
        base_url=base_url,
        request_id=request_id,
        username=username,
        password=password,
        connect_timeout=connect_timeout,
        request_timeout=request_timeout,
    )

    upload_text = (
        "CONFIDENTIAL DRAFT - PARALEGAL WORK PRODUCT\n\n"
        "Franklin County Superior Court\n"
        "Riley v. Northstar Home Robotics, Inc.\n\n"
        "The uploaded draft states that Dr. Lena Marquez did not test the Northstar Model H-17 "
        "battery pack under substantially similar charging conditions. "
        "The strongest motion argument is that Dr. Marquez failed to connect her observations "
        "to a reliable causation methodology and failed to account for the extension cord and "
        "wall receptacle as alternative ignition sources."
    )
    upload_response = _live_request(
        base_url,
        request_id,
        "POST",
        "/api/uploads",
        token=token,
        timeout=(connect_timeout, query_timeout),
        data={"is_privileged": "true"},
        files={
            "files": (
                "northstar_upload_probe.txt",
                BytesIO(upload_text.encode("utf-8")),
                "text/plain",
            )
        },
    )
    assert upload_response.status_code == 201, upload_response.text
    upload_payload = upload_response.json()
    assert upload_payload["files_uploaded"] == 1
    assert upload_payload["chunks_upserted"] >= 1
    assert upload_payload["files"][0]["filename"] == "northstar_upload_probe.txt"

    query = (
        "Using my uploaded Northstar Model H-17 draft, what is the strongest motion "
        "argument about Dr. Lena Marquez's causation opinion?"
    )
    query_response = _live_request(
        base_url,
        request_id,
        "POST",
        "/api/query",
        token=token,
        timeout=(connect_timeout, query_timeout),
        json={
            "query": query,
            "conversation_id": None,
            "include_uploaded_chunks": True,
        },
    )
    assert query_response.ok, query_response.text

    payload = query_response.json()
    answer = payload["assistant_message"]["content"]
    normalized_answer = " ".join(answer.split())
    pipeline = payload["pipeline"]

    assert "Dr. Lena Marquez" in normalized_answer
    assert "reliable causation methodology" in normalized_answer or "alternative ignition sources" in normalized_answer
    assert pipeline["user_upload_retrieval_backend_status"] == "ok"
    assert pipeline["user_upload_retrieval_chunk_count"] >= 1
    assert pipeline["prompt_case_filter_status"] == "applied:user_upload_only"
    assert pipeline["generation_context_status"] == "applied:user_upload_context"
    assert pipeline["verification_backend_status"] in {
        "ok",
        "warning:fallback:HeuristicNLIVerifier",
    }
    assert pipeline["claim_support_summary"]["total"] >= 1
    assert pipeline["claim_support_summary"]["unsupported_ratio"] < 0.5
    assert any(
        chunk["doc_type"] == "user_upload"
        and chunk["source_file"] == "northstar_upload_probe.txt"
        for chunk in pipeline["retrieved_chunks"]
    )


@pytest.mark.skipif(not _live_base_url(), reason="LIVE_API_BASE_URL is not set.")
def test_live_http_user_upload_comparison_query_keeps_public_corpus_context():
    base_url = _live_base_url()
    assert base_url is not None

    username = f"live_upload_compare_{uuid4().hex[:8]}"
    password = "live_probe_password_123"
    connect_timeout, request_timeout, query_timeout = _live_timeouts()
    request_id = uuid4().hex
    token = _register_or_login_live_user(
        base_url=base_url,
        request_id=request_id,
        username=username,
        password=password,
        connect_timeout=connect_timeout,
        request_timeout=request_timeout,
    )

    upload_text = (
        "CONFIDENTIAL DRAFT - PARALEGAL WORK PRODUCT\n\n"
        "Franklin County Superior Court\n"
        "Riley v. Northstar Home Robotics, Inc.\n\n"
        "The uploaded draft states that Dr. Lena Marquez did not test the Northstar Model H-17 "
        "battery pack under substantially similar charging conditions. "
        "The strongest motion argument is that Dr. Marquez failed to connect her observations "
        "to a reliable causation methodology and failed to account for the extension cord and "
        "wall receptacle as alternative ignition sources."
    )
    upload_response = _live_request(
        base_url,
        request_id,
        "POST",
        "/api/uploads",
        token=token,
        timeout=(connect_timeout, query_timeout),
        data={"is_privileged": "true"},
        files={
            "files": (
                "northstar_upload_comparison_probe.txt",
                BytesIO(upload_text.encode("utf-8")),
                "text/plain",
            )
        },
    )
    assert upload_response.status_code == 201, upload_response.text
    upload_payload = upload_response.json()
    assert upload_payload["files_uploaded"] == 1
    assert upload_payload["chunks_upserted"] >= 1

    query = (
        "Compare my uploaded Northstar Model H-17 draft to relevant corpus cases or precedent. "
        "How does the argument about Dr. Lena Marquez's causation methodology compare to prior cases?"
    )
    query_response = _live_request(
        base_url,
        request_id,
        "POST",
        "/api/query",
        token=token,
        timeout=(connect_timeout, query_timeout),
        json={"query": query, "conversation_id": None},
    )
    assert query_response.ok, query_response.text

    payload = query_response.json()
    answer = payload["assistant_message"]["content"]
    pipeline = payload["pipeline"]
    retrieved_chunks = pipeline["retrieved_chunks"]
    prompt_chunk_ids = set(pipeline["prompt_chunk_ids"])
    upload_chunk_ids = {
        chunk["id"]
        for chunk in retrieved_chunks
        if chunk["doc_type"] == "user_upload"
        and chunk["source_file"] == "northstar_upload_comparison_probe.txt"
    }
    public_chunk_ids = {
        chunk["id"] for chunk in retrieved_chunks if chunk["doc_type"] != "user_upload"
    }

    assert "Dr. Lena Marquez" in answer
    assert pipeline["user_upload_retrieval_backend_status"] == "ok"
    assert pipeline["user_upload_retrieval_chunk_count"] >= 1
    assert pipeline["prompt_case_filter_status"] == "applied:user_upload_with_comparison_authorities"
    assert pipeline["generation_context_status"] == "applied:user_upload_context"
    assert pipeline["public_retrieval_chunk_count"] >= 1
    assert pipeline["prompt_chunk_count"] >= 2
    assert prompt_chunk_ids & upload_chunk_ids
    assert prompt_chunk_ids & public_chunk_ids
    assert pipeline["verification_backend_status"] in {
        "ok",
        "warning:fallback:HeuristicNLIVerifier",
    }
    assert pipeline["claim_support_summary"]["total"] >= 1

    public_rerank_meta = pipeline.get("public_rerank_meta") or {}
    assert public_rerank_meta["status"] == "not_applied:disabled_query_variant_only"
    assert pipeline["public_retrieval_query_meta"]["status"] == "applied:user_upload_comparison_rewrite"
    assert "Daubert" in pipeline["public_retrieval_query"]
    assert "expert opinion admissibility" in pipeline["public_retrieval_query"]
