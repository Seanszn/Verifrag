"""Probe a running backend over real HTTP using the same flow as the frontend."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any
from uuid import uuid4

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--username", default=f"probe_{uuid4().hex[:8]}")
    parser.add_argument("--password", default="probe_password_123")
    parser.add_argument("--query", default="test")
    parser.add_argument("--conversation-id", type=int, default=None)
    parser.add_argument("--connect-timeout", type=int, default=10)
    parser.add_argument("--request-timeout", type=int, default=30)
    parser.add_argument("--query-timeout", type=int, default=180)
    parser.add_argument("--register-if-missing", action="store_true")
    parser.add_argument("--fetch-details", action="store_true")
    return parser.parse_args()


def make_request(
    method: str,
    base_url: str,
    path: str,
    *,
    token: str | None = None,
    timeout: tuple[int, int],
    request_id: str,
    **kwargs: Any,
) -> requests.Response:
    headers = {"X-Request-ID": request_id}
    if "json" in kwargs:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"

    started = time.perf_counter()
    response = requests.request(
        method,
        f"{base_url.rstrip('/')}{path}",
        headers=headers,
        timeout=timeout,
        **kwargs,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(
        json.dumps(
            {
                "path": path,
                "method": method.upper(),
                "status_code": response.status_code,
                "elapsed_ms": round(elapsed_ms, 1),
                "request_id": request_id,
            }
        )
    )
    return response


def extract_json(response: requests.Response) -> dict[str, Any] | list[Any] | None:
    try:
        return response.json()
    except ValueError:
        return None


def main() -> int:
    args = parse_args()
    api_timeout = (args.connect_timeout, args.request_timeout)
    query_timeout = (args.connect_timeout, args.query_timeout)
    request_id = uuid4().hex

    token: str | None = None
    auth_response = make_request(
        "POST",
        args.base_url,
        "/api/auth/login",
        timeout=api_timeout,
        request_id=request_id,
        json={"username": args.username, "password": args.password},
    )
    if auth_response.status_code == 401 and args.register_if_missing:
        auth_response = make_request(
            "POST",
            args.base_url,
            "/api/auth/register",
            timeout=api_timeout,
            request_id=request_id,
            json={"username": args.username, "password": args.password},
        )

    if not auth_response.ok:
        payload = extract_json(auth_response)
        print(json.dumps({"auth_error": payload or auth_response.text}, indent=2))
        return 1

    auth_payload = extract_json(auth_response)
    if not isinstance(auth_payload, dict) or "token" not in auth_payload:
        print(json.dumps({"auth_error": "Unexpected auth payload", "payload": auth_payload}, indent=2))
        return 1

    token = str(auth_payload["token"])

    conversations_response = make_request(
        "GET",
        args.base_url,
        "/api/conversations",
        token=token,
        timeout=api_timeout,
        request_id=request_id,
    )
    if not conversations_response.ok:
        print(json.dumps({"conversation_error": conversations_response.text}, indent=2))
        return 1

    try:
        query_response = make_request(
            "POST",
            args.base_url,
            "/api/query",
            token=token,
            timeout=query_timeout,
            request_id=request_id,
            json={"query": args.query, "conversation_id": args.conversation_id},
        )
    except requests.Timeout:
        print(
            json.dumps(
                {
                    "query_timeout": {
                        "request_id": request_id,
                        "query": args.query,
                        "timeout_seconds": args.query_timeout,
                    }
                },
                indent=2,
            )
        )
        return 2

    payload = extract_json(query_response)
    if not query_response.ok:
        print(json.dumps({"query_error": payload or query_response.text}, indent=2))
        return 1

    if not isinstance(payload, dict):
        print(json.dumps({"query_error": "Unexpected query payload", "payload": payload}, indent=2))
        return 1

    pipeline = payload.get("pipeline", {})
    summary = {
        "request_id": request_id,
        "conversation_id": payload.get("conversation", {}).get("id"),
        "interaction_id": payload.get("interaction", {}).get("id"),
        "llm_status": pipeline.get("llm_backend_status"),
        "retrieval_status": pipeline.get("retrieval_backend_status"),
        "verification_status": pipeline.get("verification_backend_status"),
        "retrieval_chunk_count": pipeline.get("retrieval_chunk_count"),
        "claim_count": pipeline.get("claim_count"),
        "timings_ms": pipeline.get("timings_ms"),
    }
    print(json.dumps(summary, indent=2))

    if args.fetch_details and summary["conversation_id"] is not None:
        conversation_id = int(summary["conversation_id"])
        make_request(
            "GET",
            args.base_url,
            f"/api/conversations/{conversation_id}/messages",
            token=token,
            timeout=api_timeout,
            request_id=request_id,
        )
        make_request(
            "GET",
            args.base_url,
            f"/api/conversations/{conversation_id}/interactions",
            token=token,
            timeout=api_timeout,
            request_id=request_id,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
