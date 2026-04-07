"""HTTP helpers for the Streamlit client."""

from __future__ import annotations

from typing import Any

import requests
import streamlit as st

from src.config import API


def api_headers() -> dict[str, str]:
    token = st.session_state.get("auth_token")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def api_request(method: str, path: str, **kwargs: Any) -> requests.Response:
    url = f"{API.client_api_base_url.rstrip('/')}{path}"
    headers = api_headers()
    extra_headers = kwargs.pop("headers", {})
    headers.update(extra_headers)
    return requests.request(method, url, headers=headers, timeout=60, **kwargs)


def load_conversations() -> list[dict[str, Any]]:
    response = api_request("GET", "/api/conversations")
    if response.status_code == 401:
        clear_auth()
        return []
    response.raise_for_status()
    return response.json()


def load_messages(conversation_id: int) -> list[dict[str, Any]]:
    response = api_request("GET", f"/api/conversations/{conversation_id}/messages")
    response.raise_for_status()
    return response.json()


def set_auth(auth_payload: dict[str, Any]) -> None:
    st.session_state["auth_token"] = auth_payload["token"]
    st.session_state["user"] = auth_payload["user"]
    st.session_state["selected_conversation_id"] = None


def clear_auth() -> None:
    st.session_state.pop("auth_token", None)
    st.session_state.pop("user", None)
    st.session_state.pop("selected_conversation_id", None)

def ask_agent(query: str, conversation_id: int | None = None) -> dict[str, Any]:
    payload = {
        "query": query,
        "conversation_id": conversation_id,
    }

    response = api_request("POST", "/api/agent/ask", json=payload)

    if response.status_code == 401:
        clear_auth()
        raise RuntimeError("Your session expired. Please sign in again.")

    response.raise_for_status()
    return response.json()

def ask_agent(query: str, conversation_id: int | None = None) -> dict[str, Any]:
    response = api_request(
        "POST",
        "/api/agent/ask",
        json={"query": query, "conversation_id": conversation_id},
    )
    response.raise_for_status()
    return response.json()