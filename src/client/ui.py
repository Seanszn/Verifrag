"""Streamlit client wired to the FastAPI backend."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Callable

import streamlit as st

from src.client.api_client import (
    APIError,
    clear_auth,
    load_conversations,
    load_messages,
    login as login_request,
    logout as logout_request,
    register as register_request,
    set_auth,
    submit_query as submit_query_request,
)


PAGE_LOGIN = "login"
PAGE_HOME = "home"
PAGE_UPLOAD = "upload"
PAGE_RESPONSE = "response"
SUPPORTED_UPLOAD_TYPES = ["pdf", "txt", "docx", "md"]


def initialize_state() -> None:
    defaults: dict[str, Any] = {
        "authenticated": False,
        "auth_token": None,
        "page": PAGE_LOGIN,
        "uploaded_files": [],
        "conversations": [],
        "selected_conversation_id": None,
        "conversation_messages": [],
        "messages_loaded_for": None,
        "conversation_refresh_needed": False,
        "current_query": "",
        "is_generating": False,
        "user": None,
        "show_register_notice": False,
        "registration_success_message": None,
        "logout_notice": None,
        "show_upload_hint": False,
        "upload_notice": None,
        "last_pipeline": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    sync_auth_state()


def sync_auth_state() -> None:
    st.session_state["authenticated"] = bool(
        st.session_state.get("auth_token") and st.session_state.get("user")
    )


def apply_styles() -> None:
    st.markdown(
        """
        <style>

        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        div[data-testid="stTextInput"],
        div[data-testid="stTextInput"] > div {
            width: 100% !important;
        }

        div[data-testid="stTextInput"] div[data-baseweb="input"],
        div[data-testid="stTextInput"] div[data-baseweb="base-input"] {
            width: 100% !important;
        }

        html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"] {
            font-family: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
        }

        code, pre, kbd {
            font-family: "IBM Plex Mono", Consolas, monospace;
        }

        [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #f6f7f9 0%, #eef1f5 100%) !important;
        }

        .block-container {
            background: #ffffff;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
            padding: 3rem 2rem !important;
            margin-top: 2rem;
            margin-bottom: 2rem;
            max-width: 1200px !important;
        }

        div[data-testid="columns"]:has(.vr-brand),
        div[data-testid="stHorizontalBlock"]:has(.vr-brand) {
            background: #121417;
            border: 1px solid #1f2937;
            border-radius: 20px;
            padding: 0.85rem 1rem;
            margin-top: -5rem !important;
            margin-bottom: 2rem;
            align-items: center;
        }

        .vr-brand {
            color: #f5f7fa;
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }

        .vr-page-title {
            color: #111827;
            font-size: 2rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.3rem;
        }

        .vr-page-title-muted {
            color: #6b7280;
        }

        .vr-page-subtitle {
            color: #4b5563;
            text-align: center;
            margin-bottom: 1.5rem;
        }

        .stTextInput label,
        .stTextArea label,
        .stFileUploader label {
            color: #161616;
            font-size: 0.95rem;
            font-weight: 600;
        }

        div[data-baseweb="input"],
        div[data-baseweb="textarea"] {
            border-radius: 12px !important;
            border: 2px solid #8d8d8d !important;
            background: #ffffff !important;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        div[data-baseweb="input"]:focus-within,
        div[data-baseweb="textarea"]:focus-within {
            border-color: #0f62fe !important;
            box-shadow: 0 0 0 1px #0f62fe !important;
        }

        .stTextInput input,
        .stTextArea textarea {
            width: 100% !important;
            box-sizing: border-box !important;
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            color: #161616;
            font-size: 1rem;
            padding-left: 0.9rem;
            outline: none !important;
        }

        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder {
            color: #6f6f6f;
        }

        .vr-login-anchor + div[data-testid="stVerticalBlock"] [data-testid="stButton"] button,
        .vr-home-anchor + div[data-testid="stVerticalBlock"] [data-testid="stButton"] button,
        .vr-upload-actions + div[data-testid="stHorizontalBlock"] [data-testid="stButton"] button,
        .vr-query-actions + div[data-testid="stHorizontalBlock"] [data-testid="stButton"] button,
        .vr-session-anchor + div[data-testid="stVerticalBlock"] [data-testid="stButton"] button {
            border-radius: 999px;
            font-weight: 600;
            min-height: 2.9rem;
        }

        div[data-testid="stButton"] button[kind="primary"] {
            background: #161616;
            color: #ffffff;
            border: 1px solid #161616;
        }

        div[data-testid="stButton"] button[kind="primary"]:hover {
            background: #6b7280 !important;
            color: #ffffff !important;
            border-color: #6b7280 !important;
        }

        div[data-testid="columns"]:has(.vr-brand) button,
        div[data-testid="stHorizontalBlock"]:has(.vr-brand) button {
            border-radius: 999px !important;
            font-weight: 600 !important;
        }

        div[data-testid="columns"]:has(.vr-brand) button:hover,
        div[data-testid="stHorizontalBlock"]:has(.vr-brand) button:hover {
            background-color: #e5e7eb !important;
            border-color: #e5e7eb !important;
            color: #111827 !important;
        }

        div[data-testid="columns"]:has(.vr-brand) button:hover *,
        div[data-testid="stHorizontalBlock"]:has(.vr-brand) button:hover * {
            color: #111827 !important;
        }

        .vr-home-anchor + div[data-testid="stVerticalBlock"] [data-testid="stButton"] button {
            min-height: 3.3rem;
        }

        .vr-panel-label {
            color: #111827;
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
        }

        .vr-muted {
            color: #6b7280;
            font-size: 0.95rem;
        }

        [data-testid="stFileUploader"] section {
            border-radius: 18px;
            border: 1.5px dashed #aeb7c4;
            background: #fbfcfe;
            padding: 1rem 0.8rem;
        }

        .vr-query-actions + div[data-testid="stHorizontalBlock"] [data-testid="stButton"]:first-child button {
            background: #f3d6d8;
            color: #7f1d1d;
            border: 1px solid #e7b9be;
        }

        .vr-session-anchor + div[data-testid="stVerticalBlock"] [data-testid="stButton"] button {
            text-align: left;
            justify-content: flex-start;
        }

        div[data-testid="InputInstructions"] {
            display: none !important;
            visibility: hidden !important;
        }

        header[data-testid="stHeader"] {
            display: none !important;
            visibility: hidden !important;
        }

        footer {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def navigate(page: str) -> None:
    st.session_state["page"] = page
    st.rerun()


def reset_conversation_state() -> None:
    st.session_state["uploaded_files"] = []
    st.session_state["conversations"] = []
    st.session_state["selected_conversation_id"] = None
    st.session_state["conversation_messages"] = []
    st.session_state["messages_loaded_for"] = None
    st.session_state["conversation_refresh_needed"] = False
    st.session_state["current_query"] = ""
    st.session_state["is_generating"] = False
    st.session_state["upload_notice"] = None
    st.session_state["last_pipeline"] = None


def logout() -> None:
    notice = None
    try:
        logout_request()
    except APIError as exc:
        notice = f"Signed out locally. Backend logout failed: {exc}"

    clear_auth()
    reset_conversation_state()
    st.session_state["page"] = PAGE_LOGIN
    st.session_state["show_register_notice"] = False
    st.session_state["registration_success_message"] = None
    st.session_state["logout_notice"] = notice
    st.rerun()


def render_header(action_label: str, action_callback: Callable[[], None]) -> None:
    brand_col, action_col = st.columns([6, 1.5], vertical_alignment="center")
    with brand_col:
        st.markdown('<div class="vr-brand">VerifiRAG</div>', unsafe_allow_html=True)
    with action_col:
        st.button(
            action_label,
            key=f"header_{st.session_state['page']}_{action_label}",
            use_container_width=True,
            on_click=action_callback,
        )


def handle_api_error(exc: APIError) -> None:
    sync_auth_state()
    if not st.session_state["authenticated"]:
        st.session_state["page"] = PAGE_LOGIN
        st.warning("Your session expired. Sign in again.")
        st.stop()
    st.error(str(exc))


def require_auth() -> None:
    sync_auth_state()
    if not st.session_state["authenticated"]:
        st.session_state["page"] = PAGE_LOGIN
        st.warning("Sign in to continue.")
        st.stop()


def render_page_intro(title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="vr-page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="vr-page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def refresh_conversations() -> None:
    conversations = load_conversations()
    st.session_state["conversations"] = conversations
    selected_id = st.session_state.get("selected_conversation_id")
    valid_ids = {item["id"] for item in conversations}
    if selected_id not in valid_ids:
        st.session_state["selected_conversation_id"] = None
        st.session_state["conversation_messages"] = []
        st.session_state["messages_loaded_for"] = None
    st.session_state["conversation_refresh_needed"] = False


def load_selected_conversation_messages(*, force: bool = False) -> None:
    conversation_id = st.session_state.get("selected_conversation_id")
    if conversation_id is None:
        st.session_state["conversation_messages"] = []
        st.session_state["messages_loaded_for"] = None
        return
    if not force and st.session_state.get("messages_loaded_for") == conversation_id:
        return

    st.session_state["conversation_messages"] = load_messages(conversation_id)
    st.session_state["messages_loaded_for"] = conversation_id


def upsert_conversation(conversation: dict[str, Any]) -> None:
    conversations = [
        item for item in st.session_state.get("conversations", []) if item.get("id") != conversation.get("id")
    ]
    conversations.insert(0, conversation)
    conversations.sort(key=lambda item: (item.get("updated_at", ""), item.get("id", 0)), reverse=True)
    st.session_state["conversations"] = conversations


def render_session_list(conversations: list[dict[str, Any]], selected_conversation_id: int | None) -> None:
    with st.container(border=True):
        st.markdown('<div class="vr-panel-label">Previous Sessions:</div>', unsafe_allow_html=True)

        if not conversations:
            st.markdown('<div class="vr-muted">No prior sessions yet.</div>', unsafe_allow_html=True)
            return

        st.markdown('<div class="vr-session-anchor"></div>', unsafe_allow_html=True)
        for conversation in conversations:
            label = conversation["title"]
            if selected_conversation_id == conversation["id"]:
                label = f"{label} - Active"
            if st.button(label, key=f"session_{conversation['id']}", use_container_width=True):
                st.session_state["selected_conversation_id"] = conversation["id"]
                st.session_state["conversation_messages"] = []
                st.session_state["messages_loaded_for"] = None
                st.session_state["last_pipeline"] = None
                st.rerun()


def render_chat_panel(messages: list[dict[str, Any]]) -> None:
    with st.container(border=True, height=430):
        if not messages:
            st.markdown(
                '<div class="vr-muted">Select a prior session or submit a new query.</div>',
                unsafe_allow_html=True,
            )
            return

        for message in messages:
            role = "assistant" if message["role"] == "assistant" else "user"
            with st.chat_message(role):
                st.markdown(message["content"])


def render_login_page() -> None:
    mode_label = "Switch to Login" if st.session_state.get("show_register_notice") else "Register"

    def toggle_mode() -> None:
        st.session_state["show_register_notice"] = not st.session_state.get("show_register_notice", False)

    render_header(mode_label, toggle_mode)

    is_registering = st.session_state.get("show_register_notice", False)

    _, form_col, _ = st.columns([1.2, 1.6, 1.2])
    with form_col:
        if st.session_state.get("registration_success_message"):
            st.success(st.session_state["registration_success_message"])
            st.session_state["registration_success_message"] = None
        if st.session_state.get("logout_notice"):
            st.info(st.session_state["logout_notice"])
            st.session_state["logout_notice"] = None

        title_text = "Create an " if is_registering else "Sign into your "
        st.markdown(
            f'<div class="vr-page-title"><span class="vr-page-title-muted">{title_text}</span>account</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="vr-login-anchor"></div>', unsafe_allow_html=True)
        username = st.text_input("Username", key="login_username", placeholder="Enter your username")
        password = st.text_input(
            "Password",
            type="password",
            key="login_password",
            placeholder="Enter your password",
        )

        st.write("")
        st.write("")

        button_label = "Register Account" if is_registering else "Sign In"

        if st.button(button_label, key="login_submit", type="primary", use_container_width=True):
            username_val = username.strip()
            if not username_val or not password:
                st.error("Please fill out both fields.")
                return

            try:
                if is_registering:
                    register_request(username_val, password)
                    st.session_state["show_register_notice"] = False
                    st.session_state["registration_success_message"] = (
                        "Registration successful! Switching to login..."
                    )
                    st.rerun()

                auth_payload = login_request(username_val, password)
                set_auth(auth_payload)
                reset_conversation_state()
                st.session_state["page"] = PAGE_HOME
                st.session_state["conversation_refresh_needed"] = True
                st.rerun()
            except APIError as exc:
                st.error(str(exc))


def render_home_page() -> None:
    require_auth()
    render_header("Log Out", logout)

    user = st.session_state.get("user") or {"username": "Client"}
    render_page_intro(
        "Client Workspace",
        f"Signed in as {user['username']}. Choose how you want to continue.",
    )

    _, middle, _ = st.columns([1.4, 1.2, 1.4])
    with middle:
        st.markdown('<div class="vr-home-anchor"></div>', unsafe_allow_html=True)
        if st.button("Query", key="home_query", type="primary", use_container_width=True):
            st.session_state["conversation_refresh_needed"] = True
            navigate(PAGE_RESPONSE)
        st.write("")
        if st.button("Upload Documents", key="home_upload", use_container_width=True):
            navigate(PAGE_UPLOAD)


def render_upload_page() -> None:
    require_auth()
    render_header("Home", lambda: navigate(PAGE_HOME))
    render_page_intro("Document Uploads", "Upload API wiring is pending in this branch.")

    icon_base64 = ""
    icon_path = Path(__file__).parent.parent.parent / "assets" / "upload.png"
    if icon_path.exists():
        icon_base64 = base64.b64encode(icon_path.read_bytes()).decode("ascii")

    st.markdown(
        f"""
        <style>
        [data-testid="stFileUploader"] {{ display: flex; justify-content: center; width: 100%; }}
        [data-testid="stFileUploader"] section {{
            position: relative; width: 100%; max-width: 500px; height: 220px;
            margin: 0 auto; border-radius: 16px; border: 2px dashed #bfc5cd;
            background: #f3f4f6; display: flex; align-items: center;
            justify-content: center; cursor: pointer; transition: all 0.2s ease-in-out;
        }}
        [data-testid="stFileUploader"] section:hover {{ border-color: #6b7280; background: #e5e7eb; }}
        [data-testid="stFileUploader"] button, [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] svg, [data-testid="stFileUploader"] p {{ display: none !important; }}
        [data-testid="stFileUploader"] section > div {{ opacity: 0; }}
        [data-testid="stFileUploader"] section::after {{
            content: ""; width: 70px; height: 70px;
            background-image: url("data:image/png;base64,{icon_base64}");
            background-size: contain; background-repeat: no-repeat;
            background-position: center; position: absolute;
        }}
        [data-testid="stFileUploader"] section::before {{
            content: "Upload Files"; position: absolute; bottom: 20px;
            font-size: 14px; color: #6b7280;
        }}
        .upload-instruction {{
            text-align: center;
            color: #6b7280;
            font-size: 0.95rem;
            margin-top: 15px;
            margin-bottom: 5px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([0.7, 2.2, 0.7])
    with center:
        uploaded_files = st.file_uploader(
            "",
            type=SUPPORTED_UPLOAD_TYPES,
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        st.markdown(
            '<p class="upload-instruction">Backend upload integration is the next change set. Files are not sent yet.</p>',
            unsafe_allow_html=True,
        )

        if uploaded_files:
            st.write("")
            st.markdown("**Selected Files:**")
            for file in uploaded_files:
                st.markdown(f"- {file.name} ({int(file.size)} bytes)")

        st.write("")
        _, button_col, _ = st.columns([1, 2, 1])
        with button_col:
            if st.button("Submit", key="submit_upload", type="primary", use_container_width=True):
                if not uploaded_files:
                    st.error("Select at least one file before submitting.")
                else:
                    st.info("Upload wiring is pending. This change set only connects auth, query, and history.")


def submit_query() -> None:
    query = st.session_state["current_query"].strip()
    if not query:
        st.error("Enter a query before submitting.")
        return

    st.session_state["is_generating"] = True
    try:
        selected_id = st.session_state.get("selected_conversation_id")
        result = submit_query_request(query, conversation_id=selected_id)
        conversation = result["conversation"]
        conversation_id = conversation["id"]

        previous_messages = (
            st.session_state["conversation_messages"]
            if st.session_state.get("messages_loaded_for") == conversation_id
            else []
        )
        st.session_state["selected_conversation_id"] = conversation_id
        st.session_state["conversation_messages"] = [
            *previous_messages,
            result["user_message"],
            result["assistant_message"],
        ]
        st.session_state["messages_loaded_for"] = conversation_id
        st.session_state["last_pipeline"] = result.get("pipeline")
        st.session_state["current_query"] = ""
        upsert_conversation(conversation)
        st.rerun()
    except APIError as exc:
        handle_api_error(exc)
    finally:
        st.session_state["is_generating"] = False


def render_pipeline_summary() -> None:
    pipeline = st.session_state.get("last_pipeline")
    if not pipeline:
        return
    with st.container(border=True):
        st.markdown('<div class="vr-panel-label">Latest Pipeline Status</div>', unsafe_allow_html=True)
        st.markdown(
            (
                f"- LLM: `{pipeline.get('llm_backend_status', 'unknown')}`\n"
                f"- Retrieval: `{pipeline.get('retrieval_backend_status', 'unknown')}`\n"
                f"- Verification: `{pipeline.get('verification_backend_status', 'unknown')}`\n"
                f"- Claims: `{pipeline.get('claim_count', 0)}`"
            )
        )


def render_response_page() -> None:
    require_auth()

    if st.session_state.get("conversation_refresh_needed"):
        try:
            refresh_conversations()
        except APIError as exc:
            handle_api_error(exc)

    if st.session_state.get("selected_conversation_id") is not None and (
        st.session_state.get("messages_loaded_for") != st.session_state.get("selected_conversation_id")
    ):
        try:
            load_selected_conversation_messages()
        except APIError as exc:
            handle_api_error(exc)

    render_header("Home", lambda: navigate(PAGE_HOME))

    left_col, right_col = st.columns([1.1, 2.4], gap="large")

    with left_col:
        render_session_list(
            st.session_state["conversations"],
            st.session_state["selected_conversation_id"],
        )
        render_pipeline_summary()

    with right_col:
        render_chat_panel(st.session_state["conversation_messages"])
        st.write("")
        st.text_area(
            "Start a Query",
            key="current_query",
            placeholder="Start a Query...",
            height=100,
        )

        st.markdown('<div class="vr-query-actions"></div>', unsafe_allow_html=True)
        stop_col, send_col = st.columns([1, 1.4])
        with stop_col:
            if st.button("Stop", key="stop_query", use_container_width=True):
                st.session_state["is_generating"] = False
                st.info("No active generation to stop.")
        with send_col:
            if st.button("Submit", key="submit_query", type="primary", use_container_width=True):
                submit_query()


def run_client_app() -> None:
    st.set_page_config(page_title="VerifRAG", layout="wide")
    initialize_state()
    apply_styles()

    if st.session_state["authenticated"] and st.session_state["page"] == PAGE_LOGIN:
        st.session_state["page"] = PAGE_HOME

    page = st.session_state["page"]

    if page == PAGE_LOGIN:
        render_login_page()
    elif page == PAGE_HOME:
        render_home_page()
    elif page == PAGE_UPLOAD:
        render_upload_page()
    elif page == PAGE_RESPONSE:
        render_response_page()
    else:
        navigate(PAGE_LOGIN)
