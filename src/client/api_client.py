"""HTTP helpers for the Streamlit client."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import requests
import streamlit as st

from src.config import API


class APIError(RuntimeError):
    """Raised when the backend request fails or returns an application error."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response: requests.Response | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = response


def api_headers(*, include_content_type: bool = True) -> dict[str, str]:
    headers: dict[str, str] = {}
    if include_content_type:
        headers["Content-Type"] = "application/json"

    token = st.session_state.get("auth_token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def api_request(
    method: str,
    path: str,
    *,
    include_content_type: bool = True,
    **kwargs: Any,
) -> requests.Response:
    url = f"{API.client_api_base_url.rstrip('/')}{path}"
    headers = api_headers(include_content_type=include_content_type)
    extra_headers = kwargs.pop("headers", {})
    headers.update(extra_headers)
    headers.setdefault("X-Request-ID", uuid4().hex)
    timeout = kwargs.pop("timeout", _request_timeout(path))

    try:
        response = requests.request(method, url, headers=headers, timeout=timeout, **kwargs)
    except requests.Timeout as exc:  # pragma: no cover - exercised through UI behavior
        raise APIError(_timeout_error_message(path, timeout)) from exc
    except requests.ConnectionError as exc:  # pragma: no cover - exercised through UI behavior
        raise APIError("Could not connect to the backend API.") from exc
    except requests.RequestException as exc:  # pragma: no cover - exercised through UI behavior
        raise APIError("The backend API request failed.") from exc

    if response.status_code == 401 and st.session_state.get("auth_token"):
        clear_auth()

    return response


def _request_timeout(path: str) -> tuple[int, int]:
    read_timeout = API.query_timeout_seconds if path == "/api/query" else API.request_timeout_seconds
    return (API.connect_timeout_seconds, read_timeout)


def _timeout_error_message(path: str, timeout: float | tuple[int, int]) -> str:
    read_timeout = timeout[1] if isinstance(timeout, tuple) else timeout
    if path == "/api/query":
        return (
            "The backend query timed out waiting for a response "
            f"after {int(read_timeout)} seconds."
        )
    return f"The backend API timed out after {int(read_timeout)} seconds."


def _extract_error_message(response: requests.Response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip()
        return text or None

    detail = payload.get("detail")
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list) and detail:
        first = detail[0]
        if isinstance(first, dict):
            message = first.get("msg")
            if isinstance(message, str) and message:
                return message

    return None


def _raise_for_api_error(response: requests.Response, default_message: str) -> None:
    if response.ok:
        return
    message = _extract_error_message(response) or default_message
    raise APIError(message, status_code=response.status_code, response=response)


def register(username: str, password: str) -> dict[str, Any]:
    response = api_request(
        "POST",
        "/api/auth/register",
        json={"username": username, "password": password},
    )
    _raise_for_api_error(response, "Registration failed.")
    return response.json()


def login(username: str, password: str) -> dict[str, Any]:
    response = api_request(
        "POST",
        "/api/auth/login",
        json={"username": username, "password": password},
    )
    _raise_for_api_error(response, "Login failed.")
    return response.json()


def logout() -> None:
    response = api_request("POST", "/api/auth/logout")
    if response.status_code == 401:
        return
    _raise_for_api_error(response, "Logout failed.")


def load_conversations() -> list[dict[str, Any]]:
    response = api_request("GET", "/api/conversations")
    _raise_for_api_error(response, "Could not load conversations.")
    return response.json()


def load_messages(conversation_id: int) -> list[dict[str, Any]]:
    response = api_request("GET", f"/api/conversations/{conversation_id}/messages")
    _raise_for_api_error(response, "Could not load conversation history.")
    return response.json()


def load_interactions(conversation_id: int) -> list[dict[str, Any]]:
    response = api_request("GET", f"/api/conversations/{conversation_id}/interactions")
    _raise_for_api_error(response, "Could not load interaction evidence.")
    return response.json()


def submit_query(query: str, conversation_id: int | None = None) -> dict[str, Any]:
    response = api_request(
        "POST",
        "/api/query",
        json={"query": query, "conversation_id": conversation_id},
    )
    _raise_for_api_error(response, "Query submission failed.")
    return response.json()


def upload_documents(
    uploaded_files: list[Any],
    *,
    conversation_id: int | None = None,
    is_privileged: bool = True,
) -> dict[str, Any]:
    files = [
        (
            "files",
            (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream",
            ),
        )
        for uploaded_file in uploaded_files
    ]
    form_data: dict[str, str] = {"is_privileged": str(is_privileged).lower()}
    if conversation_id is not None:
        form_data["conversation_id"] = str(conversation_id)

    response = api_request(
        "POST",
        "/api/uploads",
        include_content_type=False,
        data=form_data,
        files=files,
    )
    _raise_for_api_error(response, "Upload failed.")
    return response.json()


def set_auth(auth_payload: dict[str, Any]) -> None:
    st.session_state["auth_token"] = auth_payload["token"]
    st.session_state["user"] = auth_payload["user"]
    st.session_state["authenticated"] = True
    st.session_state["selected_conversation_id"] = None


def clear_auth() -> None:
    st.session_state.pop("auth_token", None)
    st.session_state.pop("user", None)
    st.session_state.pop("selected_conversation_id", None)
    st.session_state["authenticated"] = False
