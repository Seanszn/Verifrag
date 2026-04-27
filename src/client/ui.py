"""Streamlit client wired to the FastAPI backend."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any, Callable

import streamlit as st

from src.client.api_client import (
    APIError,
    clear_auth,
    delete_conversation as delete_conversation_request,
    delete_upload,
    load_conversations,
    load_interactions,
    load_messages,
    list_uploads,
    login as login_request,
    logout as logout_request,
    register as register_request,
    set_auth,
    submit_query as submit_query_request,
    upload_documents as upload_documents_request,
)
from src.client.claim_analysis import (
    SUPPORT_LEVEL_ORDER,
    extract_claim_evaluations,
    find_cases_for_interaction,
    group_evidence_case_references_by_support_level,
)
from src.verification.claim_contract import (
    attach_claim_citation_links,
    build_claim_citation_links,
    support_level_from_verdict,
)


PAGE_LOGIN = "login"
PAGE_HOME = "home"
PAGE_UPLOAD = "upload"
PAGE_RESPONSE = "response"
SUPPORTED_UPLOAD_TYPES = ["pdf", "txt", "docx", "md"]
SUPPORT_LEVEL_DISPLAY = {
    "supported": "Supported",
    "possibly_supported": "Possibly Supported",
    "unsupported": "Unsupported",
}
SUPPORT_LEVEL_CSS_CLASS = {
    "supported": "vr-claim-supported",
    "possibly_supported": "vr-claim-possibly-supported",
    "unsupported": "vr-claim-unsupported",
}
PROGRESS_STAGE_LABELS = {
    "queued": "Prompt submitted",
    "backend_processing": "Backend processing",
    "retrieval": "Legal context retrieval",
    "generation": "Answer generation",
    "claim_decomposition": "Claim extraction",
    "verification": "Claim verification",
    "complete": "Response ready",
    "error": "Request failed",
}


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
        "include_uploaded_chunks": False,
        "is_generating": False,
        "pending_query": None,
        "query_progress": None,
        "clear_query_after_submit": False,
        "user": None,
        "show_register_notice": False,
        "registration_success_message": None,
        "logout_notice": None,
        "show_upload_hint": False,
        "upload_notice": None,
        "last_pipeline": None,
        "query_error": None,
        "interaction_details": {},
        "interaction_case_analysis": {},
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

        .vr-logo-link {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 3rem;
            height: 3rem;
            background: #ffffff;
            border-radius: 999px;
            padding: 0.42rem;
            box-sizing: border-box;
            transform: translateY(-0.24rem);
        }

        .vr-logo-link img {
            display: block;
            width: 92%;
            height: 92%;
            object-fit: contain;
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

        .vr-progress-panel {
            border: 1px solid #d8dee8;
            border-radius: 18px;
            background: linear-gradient(180deg, #fbfcff 0%, #f3f6fa 100%);
            padding: 1rem;
            margin-bottom: 1rem;
        }

        .vr-progress-kicker {
            color: #5b6472;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            margin-bottom: 0.3rem;
            text-transform: uppercase;
        }

        .vr-progress-current {
            color: #111827;
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .vr-progress-detail {
            color: #4b5563;
            font-size: 0.88rem;
            line-height: 1.45;
            margin-bottom: 0.8rem;
        }

        .vr-progress-step {
            display: grid;
            grid-template-columns: 1.4rem 1fr;
            gap: 0.55rem;
            margin-top: 0.65rem;
        }

        .vr-progress-dot {
            align-items: center;
            border-radius: 999px;
            display: inline-flex;
            font-size: 0.68rem;
            font-weight: 700;
            height: 1.2rem;
            justify-content: center;
            margin-top: 0.12rem;
            width: 1.2rem;
        }

        .vr-progress-done .vr-progress-dot {
            background: #d9f2e3;
            color: #166534;
        }

        .vr-progress-active .vr-progress-dot {
            background: #dbeafe;
            color: #1d4ed8;
        }

        .vr-progress-waiting .vr-progress-dot,
        .vr-progress-skipped .vr-progress-dot {
            background: #eef2f7;
            color: #6b7280;
        }

        .vr-progress-error .vr-progress-dot {
            background: #fde2e1;
            color: #991b1b;
        }

        .vr-progress-step-title {
            color: #111827;
            font-size: 0.88rem;
            font-weight: 700;
        }

        .vr-progress-step-meta {
            color: #6b7280;
            font-size: 0.78rem;
            line-height: 1.35;
        }

        .vr-file-chip-row {
            margin-top: 0.5rem;
        }

        .vr-file-chip {
            display: inline-block;
            background: #eef2f7;
            border: 1px solid #d6dbe4;
            border-radius: 999px;
            color: #1f2937;
            font-size: 0.85rem;
            margin: 0 0.45rem 0.45rem 0;
            padding: 0.35rem 0.8rem;
        }

        .vr-annotated-response {
            color: #111827;
            font-size: 1rem;
            line-height: 1.75;
            white-space: normal;
            word-break: break-word;
        }

        .vr-claim-highlight {
            border-radius: 0.45rem;
            padding: 0.08rem 0.2rem;
            box-decoration-break: clone;
            -webkit-box-decoration-break: clone;
            transition: background-color 0.2s ease, box-shadow 0.2s ease;
        }

        .vr-claim-supported {
            background: #d9f2e3;
            box-shadow: inset 0 -1px 0 #2f855a;
        }

        .vr-claim-possibly-supported {
            background: #fff0c7;
            box-shadow: inset 0 -1px 0 #c48a00;
        }

        .vr-claim-unsupported {
            background: #fde2e1;
            box-shadow: inset 0 -1px 0 #b83232;
        }

        .vr-claim-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.7rem;
        }

        .vr-claim-pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            border: 1px solid #d1d5db;
            color: #374151;
            font-size: 0.78rem;
            font-weight: 600;
            padding: 0.18rem 0.62rem;
        }

        .vr-claim-support-summary + div[data-testid="stHorizontalBlock"] {
            align-items: center;
            gap: 0.45rem;
            margin-top: 0.7rem;
        }

        .vr-claim-support-summary + div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            flex: 0 0 auto !important;
            min-width: fit-content !important;
            width: auto !important;
        }

        .vr-claim-support-summary + div[data-testid="stHorizontalBlock"] [data-testid="stPopover"] button,
        .vr-claim-support-summary + div[data-testid="stHorizontalBlock"] [data-testid="stButton"] button {
            align-items: center;
            border: 1px solid #d1d5db !important;
            border-radius: 999px !important;
            color: #374151 !important;
            display: inline-flex;
            font-size: 0.78rem;
            font-weight: 600;
            height: auto;
            justify-content: center;
            line-height: 1.25;
            min-height: 0;
            padding: 0.18rem 0.62rem;
            width: max-content;
        }

        .vr-claim-support-summary + div[data-testid="stHorizontalBlock"] [data-testid="stPopover"] button svg {
            display: none;
        }

        .vr-claim-support-summary + div[data-testid="stHorizontalBlock"] .vr-claim-popover-supported + div[data-testid="stPopover"] button,
        .vr-claim-support-summary + div[data-testid="stHorizontalBlock"] .vr-claim-popover-supported + div[data-testid="stButton"] button {
            background: #d9f2e3 !important;
            box-shadow: inset 0 -1px 0 #2f855a !important;
        }

        .vr-claim-support-summary + div[data-testid="stHorizontalBlock"] .vr-claim-popover-possibly-supported + div[data-testid="stPopover"] button,
        .vr-claim-support-summary + div[data-testid="stHorizontalBlock"] .vr-claim-popover-possibly-supported + div[data-testid="stButton"] button {
            background: #fff0c7 !important;
            box-shadow: inset 0 -1px 0 #c48a00 !important;
        }

        .vr-claim-support-summary + div[data-testid="stHorizontalBlock"] .vr-claim-popover-unsupported + div[data-testid="stPopover"] button,
        .vr-claim-support-summary + div[data-testid="stHorizontalBlock"] .vr-claim-popover-unsupported + div[data-testid="stButton"] button {
            background: #fde2e1 !important;
            box-shadow: inset 0 -1px 0 #b83232 !important;
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
            height: 2.9rem;
            max-height: 2.9rem;
            overflow: hidden;
        }

        .vr-session-delete-anchor + div[data-testid="stButton"] button {
            text-align: center !important;
            justify-content: center !important;
            height: 2.9rem !important;
            min-height: 2.9rem !important;
            max-height: 2.9rem !important;
            padding: 0 !important;
        }

        .vr-session-delete-anchor + div[data-testid="stButton"] button p {
            display: none !important;
        }

        .vr-session-delete-anchor + div[data-testid="stButton"] button span {
            margin: 0 !important;
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


def set_page(page: str) -> None:
    st.session_state["page"] = page


def navigate(page: str) -> None:
    set_page(page)
    st.rerun()


def reset_conversation_state() -> None:
    st.session_state["uploaded_files"] = []
    st.session_state["conversations"] = []
    st.session_state["selected_conversation_id"] = None
    st.session_state["conversation_messages"] = []
    st.session_state["messages_loaded_for"] = None
    st.session_state["conversation_refresh_needed"] = False
    st.session_state["current_query"] = ""
    st.session_state["include_uploaded_chunks"] = False
    st.session_state["is_generating"] = False
    st.session_state["pending_query"] = None
    st.session_state["query_progress"] = None
    st.session_state["clear_query_after_submit"] = False
    st.session_state["upload_notice"] = None
    st.session_state["last_pipeline"] = None
    st.session_state["query_error"] = None
    st.session_state["interaction_details"] = {}
    st.session_state["interaction_case_analysis"] = {}


def start_new_conversation() -> None:
    st.session_state["uploaded_files"] = []
    st.session_state["selected_conversation_id"] = None
    st.session_state["conversation_messages"] = []
    st.session_state["messages_loaded_for"] = None
    st.session_state["current_query"] = ""
    st.session_state["include_uploaded_chunks"] = False
    st.session_state["is_generating"] = False
    st.session_state["pending_query"] = None
    st.session_state["query_progress"] = None
    st.session_state["clear_query_after_submit"] = False
    st.session_state["upload_notice"] = None
    st.session_state["last_pipeline"] = None
    st.session_state["query_error"] = None
    st.rerun()


def delete_conversation(conversation_id: int) -> None:
    delete_conversation_request(conversation_id)
    st.session_state["conversations"] = [
        item for item in st.session_state.get("conversations", []) if item.get("id") != conversation_id
    ]
    if st.session_state.get("selected_conversation_id") == conversation_id:
        st.session_state["selected_conversation_id"] = None
        st.session_state["conversation_messages"] = []
        st.session_state["messages_loaded_for"] = None
        st.session_state["last_pipeline"] = None
    st.session_state["query_error"] = None
    st.rerun()


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
    logo_base64 = ""
    logo_path = Path(__file__).parent.parent.parent / "assets" / "VR_logo.png"
    if logo_path.exists():
        logo_base64 = base64.b64encode(logo_path.read_bytes()).decode("ascii")

    logo_col, brand_col, action_col = st.columns([0.5, 5.5, 1.5], vertical_alignment="center")
    with logo_col:
        st.markdown(
            f"""
            <a class="vr-logo-link" href="?vr_page={PAGE_HOME}" target="_self" aria-label="Home">
                <img src="data:image/png;base64,{logo_base64}" alt="VerifiRAG logo">
            </a>
            """,
            unsafe_allow_html=True,
        )
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


def format_uploaded_file_summary(file_summary: dict[str, Any]) -> str:
    filename = str(file_summary.get("filename") or file_summary.get("name") or "uploaded file")
    size_bytes = file_summary.get("size_bytes", file_summary.get("size"))
    chunk_count = file_summary.get("chunk_count")
    detail_parts: list[str] = []

    if isinstance(size_bytes, (int, float)):
        detail_parts.append(f"{int(size_bytes)} bytes")
    if isinstance(chunk_count, (int, float)):
        detail_parts.append(f"{int(chunk_count)} chunks")
    if file_summary.get("is_privileged") is True:
        detail_parts.append("privileged")

    if not detail_parts:
        return f"- {filename}"
    return f"- {filename} ({', '.join(detail_parts)})"


def format_uploaded_file_chip(file_summary: dict[str, Any]) -> str:
    filename = str(file_summary.get("filename") or file_summary.get("name") or "uploaded file")
    return f'<span class="vr-file-chip">{html.escape(filename)}</span>'


def render_uploaded_file_chip_row(file_summaries: list[dict[str, Any]]) -> None:
    if not file_summaries:
        return
    chip_markup = "".join(format_uploaded_file_chip(item) for item in file_summaries)
    st.markdown(f'<div class="vr-file-chip-row">{chip_markup}</div>', unsafe_allow_html=True)


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

    messages = load_messages(conversation_id)
    cache_interaction_details(load_interactions(conversation_id))
    st.session_state["conversation_messages"] = enrich_messages_with_interaction_data(messages)
    st.session_state["messages_loaded_for"] = conversation_id
    st.session_state["query_error"] = None


def cache_interaction_details(interaction_details: list[dict[str, Any]]) -> None:
    for interaction_detail in interaction_details:
        cache_interaction_detail(interaction_detail)


def cache_interaction_detail(interaction_detail: dict[str, Any] | None) -> None:
    if not isinstance(interaction_detail, dict):
        return
    interaction = interaction_detail.get("interaction")
    if not isinstance(interaction, dict):
        return
    interaction_id = interaction.get("id")
    if not isinstance(interaction_id, int):
        return

    st.session_state.setdefault("interaction_details", {})[interaction_id] = interaction_detail
    st.session_state.setdefault("interaction_case_analysis", {})[interaction_id] = find_cases_for_interaction(
        interaction_detail
    )


def build_interaction_detail_from_query_result(result: dict[str, Any]) -> dict[str, Any] | None:
    interaction = result.get("interaction")
    if not isinstance(interaction, dict):
        return None

    pipeline = result.get("pipeline")
    if not isinstance(pipeline, dict):
        pipeline = {}

    claims = pipeline.get("claims")
    citations = pipeline.get("retrieved_chunks")
    contradictions = pipeline.get("contradictions")
    normalized_claims = claims if isinstance(claims, list) else []
    normalized_citations = citations if isinstance(citations, list) else []
    claim_citation_links = pipeline.get("claim_citation_links")
    if not isinstance(claim_citation_links, list):
        claim_citation_links = build_claim_citation_links(
            normalized_claims,
            normalized_citations,
        )

    return {
        "interaction": interaction,
        "claims": attach_claim_citation_links(normalized_claims, claim_citation_links),
        "citations": normalized_citations,
        "claim_citation_links": claim_citation_links,
        "contradictions": contradictions if isinstance(contradictions, list) else [],
    }


def enrich_messages_with_interaction_data(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    interaction_details = st.session_state.get("interaction_details", {})
    interaction_case_analysis = st.session_state.get("interaction_case_analysis", {})
    enriched_messages: list[dict[str, Any]] = []

    for message in messages:
        enriched_message = dict(message)
        interaction_id = enriched_message.get("interaction_id")
        if enriched_message.get("role") == "assistant" and isinstance(interaction_id, int):
            interaction_detail = interaction_details.get(interaction_id)
            if isinstance(interaction_detail, dict):
                enriched_message["claim_evaluations"] = extract_claim_evaluations(interaction_detail)
                enriched_message["claim_case_analysis"] = interaction_case_analysis.get(interaction_id)
            else:
                enriched_message["claim_evaluations"] = []
                enriched_message["claim_case_analysis"] = None
        enriched_messages.append(enriched_message)

    return enriched_messages


def build_annotated_response_html(
    response_text: str,
    claims: list[dict[str, Any]] | None,
    answer_blocks: list[dict[str, Any]] | None = None,
    *,
    include_legend: bool = True,
) -> str | None:
    text = str(response_text or "")
    if not text:
        return None

    highlights = _resolve_response_claim_highlights(text, claims or [], answer_blocks or [])
    if not highlights:
        return None

    segments: list[str] = []
    cursor = 0
    for highlight in highlights:
        start = highlight["start"]
        end = highlight["end"]
        if start > cursor:
            segments.append(_escape_response_html(text[cursor:start]))
        css_class = SUPPORT_LEVEL_CSS_CLASS[highlight["support_level"]]
        tooltip = _claim_tooltip(highlight["claim"], highlight["support_level"])
        segments.append(
            (
                f'<span class="vr-claim-highlight {css_class}" '
                f'data-support-level="{highlight["support_level"]}" '
                f'title="{tooltip}">'
                f"{_escape_response_html(text[start:end])}"
                "</span>"
            )
        )
        cursor = end

    if cursor < len(text):
        segments.append(_escape_response_html(text[cursor:]))

    legend = _build_claim_legend_html(highlights) if include_legend else ""
    return f'<div class="vr-annotated-response">{"".join(segments)}</div>{legend}'


def _resolve_response_claim_highlights(
    response_text: str,
    claims: list[dict[str, Any]],
    answer_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    highlights = _resolve_claim_highlights(response_text, claims)
    answer_block_highlights = _resolve_answer_block_highlights(response_text, answer_blocks)
    if answer_block_highlights:
        highlights = _select_claim_highlights([*highlights, *answer_block_highlights])
    return highlights


def _resolve_claim_highlights(
    response_text: str,
    claims: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        span = _locate_claim_span(response_text, claim)
        if span is None:
            continue
        start, end = span
        support_level = _claim_support_level(claim)
        candidates.append(
            {
                "claim": claim,
                "start": start,
                "end": end,
                "support_level": support_level,
            }
        )

    return _select_claim_highlights(candidates)


def _resolve_answer_block_highlights(
    response_text: str,
    answer_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for block in answer_blocks:
        if not isinstance(block, dict):
            continue
        if block.get("verification_required") is not True:
            continue
        support_level = block.get("support_level")
        if support_level not in SUPPORT_LEVEL_DISPLAY:
            continue

        start = block.get("start_char")
        end = block.get("end_char")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if not 0 <= start < end <= len(response_text):
            continue

        block_text = str(block.get("text") or "")
        if block_text and response_text[start:end].strip() != block_text.strip():
            continue

        candidates.append(
            {
                "claim": _claim_payload_from_answer_block(block),
                "start": start,
                "end": end,
                "support_level": str(support_level),
            }
        )

    return _select_claim_highlights(candidates)


def _claim_payload_from_answer_block(block: dict[str, Any]) -> dict[str, Any]:
    block_type = str(block.get("type") or "answer_block")
    explanation = str(
        block.get("explanation")
        or "This answer segment requires verification."
    )
    return {
        "claim_id": block.get("block_id") or block_type,
        "text": block.get("text") or "",
        "claim_type": block_type,
        "annotation": {
            "support_level": block.get("support_level") or "unsupported",
            "explanation": explanation,
            "response_span": {
                "start_char": block.get("start_char"),
                "end_char": block.get("end_char"),
                "text": block.get("text") or "",
            },
            "evidence": [],
        },
    }


def _select_claim_highlights(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates.sort(key=lambda item: (item["start"], item["end"]))
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        if not selected or candidate["start"] >= selected[-1]["end"]:
            selected.append(candidate)
            continue
        if _highlight_priority(candidate) < _highlight_priority(selected[-1]):
            selected[-1] = candidate
    return selected


def _highlight_priority(candidate: dict[str, Any]) -> tuple[int, int, int, int, int]:
    claim = candidate["claim"]
    claim_type = str(claim.get("claim_type") or "")
    support_level = candidate["support_level"]
    span_length = candidate["end"] - candidate["start"]
    support_priority = {
        "supported": 0,
        "possibly_supported": 1,
        "unsupported": 2,
    }[support_level]
    claim_type_priority = 1 if claim_type == "attribution" else 0
    return (
        claim_type_priority,
        span_length,
        support_priority,
        candidate["start"],
        candidate["end"],
    )


def _claim_support_level(claim: dict[str, Any]) -> str:
    annotation = claim.get("annotation")
    if isinstance(annotation, dict):
        support_level = annotation.get("support_level")
        if support_level in SUPPORT_LEVEL_DISPLAY:
            return str(support_level)

    verification = claim.get("verification")
    if isinstance(verification, dict):
        return support_level_from_verdict(verification.get("verdict"))
    return "unsupported"


def _locate_claim_span(response_text: str, claim: dict[str, Any]) -> tuple[int, int] | None:
    annotation = claim.get("annotation") if isinstance(claim, dict) else None
    if not isinstance(annotation, dict):
        annotation = {}

    response_span = annotation.get("response_span")
    if not isinstance(response_span, dict):
        response_span = claim.get("span") if isinstance(claim.get("span"), dict) else {}

    preferred_start = response_span.get("start_char")
    preferred_end = response_span.get("end_char")
    claim_text = str(response_span.get("text") or claim.get("text") or "")

    if isinstance(preferred_start, int) and isinstance(preferred_end, int):
        if 0 <= preferred_start < preferred_end <= len(response_text):
            if response_text[preferred_start:preferred_end] == claim_text:
                return preferred_start, preferred_end
            if response_text[preferred_start:preferred_end].strip() == claim_text.strip():
                return preferred_start, preferred_end

    if not claim_text:
        return None

    exact_matches = _find_occurrences(response_text, claim_text)
    if exact_matches:
        return _select_best_match(exact_matches, preferred_start)

    stripped_text = claim_text.strip()
    if stripped_text and stripped_text != claim_text:
        stripped_matches = _find_occurrences(response_text, stripped_text)
        if stripped_matches:
            return _select_best_match(stripped_matches, preferred_start)

    lower_matches = _find_occurrences(response_text.lower(), claim_text.lower())
    if lower_matches:
        return _select_best_match(lower_matches, preferred_start)

    return None


def _find_occurrences(haystack: str, needle: str) -> list[tuple[int, int]]:
    if not haystack or not needle:
        return []

    matches: list[tuple[int, int]] = []
    start = 0
    while True:
        index = haystack.find(needle, start)
        if index < 0:
            break
        matches.append((index, index + len(needle)))
        start = index + 1
    return matches


def _select_best_match(
    matches: list[tuple[int, int]],
    preferred_start: Any,
) -> tuple[int, int]:
    if isinstance(preferred_start, int):
        return min(matches, key=lambda item: abs(item[0] - preferred_start))
    return matches[0]


def _claim_tooltip(claim: dict[str, Any], support_level: str) -> str:
    annotation = claim.get("annotation")
    if not isinstance(annotation, dict):
        annotation = {}
    verification = claim.get("verification")
    if not isinstance(verification, dict):
        verification = {}

    explanation = annotation.get("explanation") or verification.get("verdict_explanation") or (
        f"{SUPPORT_LEVEL_DISPLAY[support_level]} claim."
    )
    evidence = annotation.get("evidence")
    if not isinstance(evidence, list):
        evidence = claim.get("linked_citations") if isinstance(claim.get("linked_citations"), list) else []

    evidence_lines: list[str] = []
    for item in evidence[:2]:
        if not isinstance(item, dict):
            continue
        relationship = str(item.get("relationship") or "").strip()
        citation = item.get("citation")
        citation_label = None
        if isinstance(citation, dict):
            citation_label = citation.get("citation") or citation.get("case_name") or citation.get("source_label")
        citation_label = citation_label or item.get("source_label")
        if citation_label:
            evidence_lines.append(f"{relationship.title()}: {citation_label}")

    parts = [SUPPORT_LEVEL_DISPLAY[support_level], str(explanation).strip()]
    parts.extend(evidence_lines)
    return html.escape(" | ".join(part for part in parts if part), quote=True)


def _build_claim_legend_html(highlights: list[dict[str, Any]]) -> str:
    counts = _count_claim_support_levels(highlights)

    pills = [
        (
            f'<span class="vr-claim-pill {SUPPORT_LEVEL_CSS_CLASS[level]}">'
            f'{counts[level]} {SUPPORT_LEVEL_DISPLAY[level]}'
            "</span>"
        )
        for level in ("supported", "possibly_supported", "unsupported")
        if counts[level] > 0
    ]
    if not pills:
        return ""
    return f'<div class="vr-claim-legend">{"".join(pills)}</div>'


def _count_claim_support_levels(highlights: list[dict[str, Any]]) -> dict[str, int]:
    counts = {level: 0 for level in SUPPORT_LEVEL_ORDER}
    for highlight in highlights:
        support_level = highlight.get("support_level")
        if support_level in counts:
            counts[str(support_level)] += 1
    return counts


def _claim_support_counts_for_response(
    response_text: str,
    claims: list[dict[str, Any]],
    answer_blocks: list[dict[str, Any]],
) -> dict[str, int]:
    if not response_text:
        return {level: 0 for level in SUPPORT_LEVEL_ORDER}
    return _count_claim_support_levels(
        _resolve_response_claim_highlights(response_text, claims, answer_blocks)
    )


def _escape_response_html(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def _message_pipeline_metadata(message: dict[str, Any]) -> dict[str, Any]:
    metadata = message.get("metadata_json")
    if isinstance(metadata, dict):
        return metadata
    if not isinstance(metadata, str) or not metadata.strip():
        return {}
    try:
        parsed = json.loads(metadata)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _message_answer_warning(message: dict[str, Any]) -> dict[str, Any] | None:
    pipeline_metadata = _message_pipeline_metadata(message)
    answer_warning = pipeline_metadata.get("answer_warning")
    return answer_warning if isinstance(answer_warning, dict) else None


def render_assistant_message(message: dict[str, Any]) -> None:
    content = str(message.get("content") or "")
    pipeline_metadata = _message_pipeline_metadata(message)
    answer_warning = pipeline_metadata.get("answer_warning")
    if isinstance(answer_warning, dict) and answer_warning.get("show"):
        warning_message = str(answer_warning.get("message") or "").strip()
        if warning_message:
            st.warning(warning_message)
    claims = message.get("claim_evaluations")
    claim_list = claims if isinstance(claims, list) else []
    answer_blocks = pipeline_metadata.get("answer_blocks")
    if not isinstance(answer_blocks, list):
        answer_blocks = []
    if claim_list or answer_blocks:
        annotated_html = build_annotated_response_html(
            content,
            claim_list,
            answer_blocks,
            include_legend=False,
        )
        if annotated_html:
            st.markdown(annotated_html, unsafe_allow_html=True)
            _render_claim_support_popovers(
                claim_list,
                _claim_support_counts_for_response(content, claim_list, answer_blocks),
                message,
            )
            return
    st.markdown(content)


def _render_claim_support_popovers(
    claims: list[dict[str, Any]],
    support_counts: dict[str, int],
    message: dict[str, Any],
) -> None:
    active_levels = [level for level in SUPPORT_LEVEL_ORDER if support_counts.get(level, 0) > 0]
    if not active_levels:
        return

    grouped_refs = group_evidence_case_references_by_support_level(claims)
    popover = getattr(st, "popover", None)
    st.markdown('<div class="vr-claim-support-summary"></div>', unsafe_allow_html=True)
    columns = st.columns(len(active_levels))

    if callable(popover):
        for column, level in zip(columns, active_levels):
            with column:
                label = f"{support_counts[level]} {SUPPORT_LEVEL_DISPLAY[level]}"
                st.markdown(
                    f'<div class="vr-claim-popover-{_support_level_css_token(level)}"></div>',
                    unsafe_allow_html=True,
                )
                popover_context = popover(label)
                with popover_context:
                    _render_claim_support_category_details(level, grouped_refs.get(level, []))
        return

    selection_key = f"claim_support_summary_{_message_widget_key_suffix(message)}"
    for column, level in zip(columns, active_levels):
        with column:
            label = f"{support_counts[level]} {SUPPORT_LEVEL_DISPLAY[level]}"
            st.markdown(
                f'<div class="vr-claim-popover-{_support_level_css_token(level)}"></div>',
                unsafe_allow_html=True,
            )
            if st.button(label, key=f"{selection_key}_{level}"):
                st.session_state[selection_key] = level

    selected_level = st.session_state.get(selection_key)
    if selected_level in active_levels:
        _render_claim_support_category_details(
            str(selected_level),
            grouped_refs.get(str(selected_level), []),
        )


def _render_claim_support_category_details(
    support_level: str,
    case_refs: list[dict[str, Any]],
) -> None:
    label = SUPPORT_LEVEL_DISPLAY[support_level]
    st.markdown(f"**{label} Evidence**")
    if not case_refs:
        st.caption(f"No case references were attached to {label.lower()} claims.")
        return

    for index, case_ref in enumerate(case_refs, start=1):
        if index > 1:
            st.divider()
        st.markdown(f"**{_case_reference_display_name(case_ref)}**")
        metadata = _case_reference_metadata(case_ref)
        if metadata:
            st.caption(" | ".join(metadata))

        claim_texts = [
            str(text).strip()
            for text in case_ref.get("claim_texts", [])
            if str(text).strip()
        ]
        if len(claim_texts) <= 1:
            claim_text = claim_texts[0] if claim_texts else str(case_ref.get("claim_text") or "").strip()
            if claim_text:
                st.markdown(f"Claim: {claim_text}")
        else:
            st.markdown("Claims:")
            for claim_text in claim_texts:
                st.markdown(f"- {claim_text}")

        evidence_quotes = [
            str(quote).strip()
            for quote in case_ref.get("evidence_quotes", [])
            if str(quote).strip()
        ]
        if not evidence_quotes and str(case_ref.get("evidence_quote") or "").strip():
            evidence_quotes = [str(case_ref.get("evidence_quote")).strip()]
        if evidence_quotes:
            st.markdown("Document chunk:")
            for quote in evidence_quotes:
                st.markdown(f'> "{html.escape(_compact_evidence_quote(quote))}"')


def _case_reference_display_name(case_ref: dict[str, Any]) -> str:
    case_name = str(case_ref.get("case_name") or "").strip()
    reporter_citation = str(case_ref.get("reporter_citation") or "").strip()
    source_label = str(case_ref.get("source_label") or "").strip()

    display_name = case_name or source_label or "Unknown case"
    if reporter_citation and reporter_citation != display_name:
        return f"{display_name} ({reporter_citation})"
    return display_name


def _compact_evidence_quote(quote: str) -> str:
    return " ".join(str(quote or "").split())


def _case_reference_metadata(case_ref: dict[str, Any]) -> list[str]:
    metadata: list[str] = []
    relationships = [
        _format_case_relationship(relationship)
        for relationship in case_ref.get("relationships", [])
        if str(relationship).strip()
    ]
    if not relationships and case_ref.get("relationship"):
        relationships = [_format_case_relationship(case_ref["relationship"])]
    if relationships:
        metadata.append(f"Relationship: {', '.join(relationships)}")

    score = case_ref.get("score")
    if isinstance(score, (int, float)):
        metadata.append(f"Score: {_format_case_score(float(score))}")
    return metadata


def _format_case_relationship(relationship: Any) -> str:
    return str(relationship).replace("_", " ").strip().title()


def _format_case_score(score: float) -> str:
    return f"{score:.3f}".rstrip("0").rstrip(".")


def _support_level_css_token(support_level: str) -> str:
    return support_level.replace("_", "-")


def _message_widget_key_suffix(message: dict[str, Any]) -> str:
    stable_id = message.get("id") or message.get("interaction_id")
    if stable_id:
        return str(stable_id)
    content = str(message.get("content") or "")
    return str(abs(hash(content)))


def upsert_conversation(conversation: dict[str, Any]) -> None:
    conversations = [
        item for item in st.session_state.get("conversations", []) if item.get("id") != conversation.get("id")
    ]
    conversations.insert(0, conversation)
    conversations.sort(key=lambda item: (item.get("updated_at", ""), item.get("id", 0)), reverse=True)
    st.session_state["conversations"] = conversations


def render_session_list(conversations: list[dict[str, Any]], selected_conversation_id: int | None) -> None:
    with st.container(border=True):
        if st.button(
            "New Conversation",
            key="start_new_conversation",
            type="primary",
            use_container_width=True,
        ):
            start_new_conversation()

        st.write("")
        st.markdown('<div class="vr-panel-label">Previous Sessions:</div>', unsafe_allow_html=True)

        if not conversations:
            st.markdown('<div class="vr-muted">No prior sessions yet.</div>', unsafe_allow_html=True)
            return

        st.markdown('<div class="vr-session-anchor"></div>', unsafe_allow_html=True)
        for conversation in conversations:
            label = conversation["title"]
            if selected_conversation_id == conversation["id"]:
                label = f"{label} - Active"
            session_col, delete_col = st.columns([4, 1], vertical_alignment="center")
            with session_col:
                if st.button(label, key=f"session_{conversation['id']}", use_container_width=True):
                    st.session_state["selected_conversation_id"] = conversation["id"]
                    st.session_state["conversation_messages"] = []
                    st.session_state["messages_loaded_for"] = None
                    st.session_state["last_pipeline"] = None
                    st.rerun()
            with delete_col:
                st.markdown('<div class="vr-session-delete-anchor"></div>', unsafe_allow_html=True)
                if st.button(
                    " ",
                    key=f"delete_session_{conversation['id']}",
                    icon=":material/delete:",
                    use_container_width=True,
                ):
                    try:
                        delete_conversation(conversation["id"])
                    except APIError as exc:
                        handle_api_error(exc)


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
                if role == "assistant":
                    render_assistant_message(message)
                else:
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
    render_header("Home", lambda: set_page(PAGE_HOME))
    render_page_intro(
        "Document Uploads",
        "Add source documents before asking questions. Files are sent to the backend API and ingested server-side.",
    )

    st.markdown("### Your Uploaded Documents")
    try:
        existing_uploads = list_uploads()
    except APIError:
        existing_uploads = []

    if not existing_uploads:
        st.info("No documents uploaded yet.")
    else:
        for doc in existing_uploads:
            doc_col1, doc_col2, doc_col3 = st.columns([3, 1, 1])
            with doc_col1:
                st.markdown(f"**{doc.get('filename', 'Unknown')}**")
                st.caption(f"Case: {doc.get('case_name', 'N/A')} | {doc.get('chunk_count', 0)} chunks")
            with doc_col2:
                pass
            with doc_col3:
                if st.button("Delete", key=f"delete_upload_{doc.get('document_id')}", use_container_width=True):
                    try:
                        delete_upload(doc.get("document_id"))
                        st.rerun()
                    except APIError as exc:
                        st.error(f"Failed to delete: {exc}")

    st.divider()

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
            '<p class="upload-instruction">Use the upload box above to drag files in or click to browse files.</p>',
            unsafe_allow_html=True,
        )
        is_privileged = st.checkbox(
            "Treat uploaded files as privileged",
            value=True,
            help="Privileged files stay in the user upload workspace on the server.",
        )

        if uploaded_files:
            st.write("")
            st.markdown("**Selected Files:**")
            render_uploaded_file_chip_row(
                [
                    {
                        "name": file.name,
                        "size": int(file.size),
                    }
                    for file in uploaded_files
                ]
            )

        st.write("")
        _, button_col, _ = st.columns([1, 2, 1])
        with button_col:
            if st.button("Submit", key="submit_upload", type="primary", use_container_width=True):
                if not uploaded_files:
                    st.error("Select at least one file before submitting.")
                else:
                    try:
                        result = upload_documents_request(
                            list(uploaded_files),
                            conversation_id=st.session_state.get("selected_conversation_id"),
                            is_privileged=is_privileged,
                        )
                        st.session_state["uploaded_files"] = result.get("files", [])
                        st.session_state["upload_notice"] = (
                            "Uploaded "
                            f"{result.get('files_uploaded', 0)} file(s) to the backend "
                            f"and ingested {result.get('chunks_upserted', 0)} chunk(s)."
                        )
                        navigate(PAGE_RESPONSE)
                    except APIError as exc:
                        handle_api_error(exc)


def _status_kind(status: str | None) -> str:
    normalized = str(status or "").strip().lower()
    if normalized.startswith("error:") or normalized == "error":
        return "error"
    if normalized.startswith("skipped:") or normalized.startswith("disabled:"):
        return "skipped"
    if normalized.startswith("warning:"):
        return "done"
    if normalized == "ok" or normalized.startswith("applied:") or normalized.startswith("not_applied:"):
        return "done"
    if not normalized or normalized == "unknown":
        return "waiting"
    return "done"


def _format_timing_ms(timings_ms: dict[str, Any], key: str) -> str:
    value = timings_ms.get(key)
    if not isinstance(value, int | float):
        return ""
    if value >= 1000:
        return f"{value / 1000:.1f}s"
    return f"{value:.0f}ms"


def build_completed_progress_steps(pipeline: dict[str, Any] | None) -> list[dict[str, str]]:
    pipeline = pipeline if isinstance(pipeline, dict) else {}
    timings_ms = pipeline.get("timings_ms")
    timings_ms = timings_ms if isinstance(timings_ms, dict) else {}
    claim_count = int(pipeline.get("claim_count") or 0)

    retrieval_status = str(pipeline.get("retrieval_backend_status") or "unknown")
    retrieval_meta = retrieval_status
    retrieval_count = pipeline.get("retrieval_chunk_count")
    if isinstance(retrieval_count, int):
        retrieval_meta = f"{retrieval_meta} | {retrieval_count} source chunk(s)"
    retrieval_timing = _format_timing_ms(timings_ms, "retrieval")
    if retrieval_timing:
        retrieval_meta = f"{retrieval_meta} | {retrieval_timing}"

    generation_status = str(pipeline.get("llm_backend_status") or "unknown")
    generation_meta = generation_status
    generation_mode = pipeline.get("generation_mode")
    if generation_mode:
        generation_meta = f"{generation_meta} | {generation_mode}"
    generation_timing = _format_timing_ms(timings_ms, "generation")
    if generation_timing:
        generation_meta = f"{generation_meta} | {generation_timing}"

    decomposition_timing = _format_timing_ms(timings_ms, "claim_decomposition")
    decomposition_meta = f"{claim_count} claim(s) extracted"
    if decomposition_timing:
        decomposition_meta = f"{decomposition_meta} | {decomposition_timing}"
    decomposition_state = "done" if claim_count > 0 else _status_kind(pipeline.get("verification_backend_status"))
    if decomposition_state == "error":
        decomposition_meta = "Claim extraction did not complete."

    verification_status = str(pipeline.get("verification_backend_status") or "unknown")
    verification_meta = verification_status
    verification_timing = _format_timing_ms(timings_ms, "verification")
    if verification_timing:
        verification_meta = f"{verification_meta} | {verification_timing}"

    return [
        {
            "label": PROGRESS_STAGE_LABELS["retrieval"],
            "state": _status_kind(retrieval_status),
            "meta": retrieval_meta,
        },
        {
            "label": PROGRESS_STAGE_LABELS["generation"],
            "state": _status_kind(generation_status),
            "meta": generation_meta,
        },
        {
            "label": PROGRESS_STAGE_LABELS["claim_decomposition"],
            "state": decomposition_state,
            "meta": decomposition_meta,
        },
        {
            "label": PROGRESS_STAGE_LABELS["verification"],
            "state": _status_kind(verification_status),
            "meta": verification_meta,
        },
    ]


def build_running_progress() -> dict[str, Any]:
    return {
        "status": "running",
        "current_label": PROGRESS_STAGE_LABELS["backend_processing"],
        "detail": (
            "The backend has the prompt and is running retrieval, generation, claim extraction, "
            "and verification. Exact server-side stage timing appears here after the response returns."
        ),
        "steps": [
            {
                "label": PROGRESS_STAGE_LABELS["queued"],
                "state": "done",
                "meta": "Prompt captured and request opened.",
            },
            {
                "label": PROGRESS_STAGE_LABELS["backend_processing"],
                "state": "active",
                "meta": "Waiting for the backend pipeline to return stage results.",
            },
            {
                "label": PROGRESS_STAGE_LABELS["complete"],
                "state": "waiting",
                "meta": "Response and verification summary will appear when ready.",
            },
        ],
    }


def build_complete_progress(pipeline: dict[str, Any] | None) -> dict[str, Any]:
    steps = build_completed_progress_steps(pipeline)
    error_steps = [step for step in steps if step["state"] == "error"]
    skipped_steps = [step for step in steps if step["state"] == "skipped"]
    claim_count = 0
    if isinstance(pipeline, dict):
        claim_count = int(pipeline.get("claim_count") or 0)

    if error_steps:
        current_label = "Completed with pipeline errors"
        detail = "The backend returned a response, but one or more pipeline stages reported an error."
    elif skipped_steps:
        current_label = "Completed with skipped stages"
        detail = "The response is ready. Some verification work was skipped because the backend reported a skip condition."
    else:
        current_label = PROGRESS_STAGE_LABELS["complete"]
        detail = f"Generation and verification completed. {claim_count} claim(s) were prepared for review."

    return {
        "status": "complete",
        "current_label": current_label,
        "detail": detail,
        "steps": [
            {
                "label": PROGRESS_STAGE_LABELS["queued"],
                "state": "done",
                "meta": "Prompt submitted.",
            },
            *steps,
        ],
    }


def build_error_progress(message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "current_label": PROGRESS_STAGE_LABELS["error"],
        "detail": message,
        "steps": [
            {
                "label": PROGRESS_STAGE_LABELS["queued"],
                "state": "done",
                "meta": "Prompt submitted.",
            },
            {
                "label": PROGRESS_STAGE_LABELS["backend_processing"],
                "state": "error",
                "meta": message,
            },
        ],
    }


def queue_query_submission() -> None:
    query = st.session_state["current_query"].strip()
    if not query:
        st.session_state["query_error"] = "Enter a query before submitting."
        return

    st.session_state["query_error"] = None
    st.session_state["is_generating"] = True
    st.session_state["pending_query"] = {
        "query": query,
        "conversation_id": st.session_state.get("selected_conversation_id"),
        "include_uploaded_chunks": bool(st.session_state.get("include_uploaded_chunks")),
    }
    st.session_state["query_progress"] = build_running_progress()


def process_pending_query() -> None:
    pending_query = st.session_state.get("pending_query")
    if not isinstance(pending_query, dict):
        return

    query = str(pending_query.get("query") or "").strip()
    if not query:
        st.session_state["pending_query"] = None
        st.session_state["is_generating"] = False
        return

    st.session_state["query_progress"] = build_running_progress()
    try:
        selected_id = pending_query.get("conversation_id")
        result = submit_query_request(
            query,
            conversation_id=selected_id,
            include_uploaded_chunks=bool(pending_query.get("include_uploaded_chunks")),
        )
        conversation = result["conversation"]
        conversation_id = conversation["id"]

        previous_messages = (
            st.session_state["conversation_messages"]
            if st.session_state.get("messages_loaded_for") == conversation_id
            else []
        )
        cache_interaction_detail(build_interaction_detail_from_query_result(result))
        st.session_state["selected_conversation_id"] = conversation_id
        st.session_state["conversation_messages"] = enrich_messages_with_interaction_data(
            [
                *previous_messages,
                result["user_message"],
                result["assistant_message"],
            ]
        )
        st.session_state["messages_loaded_for"] = conversation_id
        st.session_state["last_pipeline"] = result.get("pipeline")
        st.session_state["query_progress"] = build_complete_progress(result.get("pipeline"))
        st.session_state["clear_query_after_submit"] = True
        upsert_conversation(conversation)
    except APIError as exc:
        st.session_state["query_error"] = str(exc)
        st.session_state["query_progress"] = build_error_progress(str(exc))
        handle_api_error(exc)
    finally:
        st.session_state["pending_query"] = None
        st.session_state["is_generating"] = False
        st.rerun()


def render_query_progress_panel() -> None:
    progress = st.session_state.get("query_progress")
    if not isinstance(progress, dict):
        with st.container(border=True):
            st.markdown('<div class="vr-panel-label">Request Progress</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="vr-muted">Submit a query to see generation and verification state here.</div>',
                unsafe_allow_html=True,
            )
        return

    status = str(progress.get("status") or "idle")
    current_label = html.escape(str(progress.get("current_label") or "Waiting"), quote=True)
    detail = html.escape(str(progress.get("detail") or ""), quote=True)
    kicker = "Live request" if status == "running" else "Latest request"
    steps = progress.get("steps")
    steps = steps if isinstance(steps, list) else []

    step_markup = []
    symbol_by_state = {
        "done": "OK",
        "active": "..",
        "waiting": "",
        "skipped": "--",
        "error": "!",
    }
    for step in steps:
        if not isinstance(step, dict):
            continue
        state = str(step.get("state") or "waiting")
        if state not in symbol_by_state:
            state = "waiting"
        label = html.escape(str(step.get("label") or "Pipeline step"), quote=True)
        meta = html.escape(str(step.get("meta") or ""), quote=True)
        symbol = html.escape(symbol_by_state[state], quote=True)
        step_markup.append(
            (
                f'<div class="vr-progress-step vr-progress-{state}">'
                f'<span class="vr-progress-dot">{symbol}</span>'
                "<div>"
                f'<div class="vr-progress-step-title">{label}</div>'
                f'<div class="vr-progress-step-meta">{meta}</div>'
                "</div>"
                "</div>"
            )
        )

    st.markdown(
        (
            '<div class="vr-progress-panel">'
            '<div class="vr-panel-label">Request Progress</div>'
            f'<div class="vr-progress-kicker">{html.escape(kicker, quote=True)}</div>'
            f'<div class="vr-progress-current">{current_label}</div>'
            f'<div class="vr-progress-detail">{detail}</div>'
            f'{"".join(step_markup)}'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


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

    if st.session_state.get("clear_query_after_submit"):
        st.session_state["current_query"] = ""
        st.session_state["clear_query_after_submit"] = False

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

    render_header("Home", lambda: set_page(PAGE_HOME))

    if st.session_state.get("upload_notice"):
        st.success(st.session_state["upload_notice"])
        st.session_state["upload_notice"] = None
    if st.session_state.get("query_error"):
        st.error(st.session_state["query_error"])

    left_col, right_col = st.columns([1.1, 2.4], gap="large")

    with left_col:
        render_query_progress_panel()
        render_session_list(
            st.session_state["conversations"],
            st.session_state["selected_conversation_id"],
        )
        render_pipeline_summary()
        if st.session_state.get("uploaded_files"):
            with st.container(border=True):
                st.markdown('<div class="vr-panel-label">Uploaded Sources</div>', unsafe_allow_html=True)
                render_uploaded_file_chip_row(st.session_state["uploaded_files"])
                for file_summary in st.session_state["uploaded_files"]:
                    st.markdown(format_uploaded_file_summary(file_summary))

    with right_col:
        render_chat_panel(st.session_state["conversation_messages"])
        st.write("")
        st.text_area(
            "Start a Query",
            key="current_query",
            placeholder="Start a Query...",
            height=100,
        )
        st.toggle(
            "Include uploaded chunks",
            key="include_uploaded_chunks",
            disabled=bool(st.session_state.get("is_generating")),
        )

        st.markdown('<div class="vr-query-actions"></div>', unsafe_allow_html=True)
        stop_col, send_col = st.columns([1, 1.4])
        with stop_col:
            if st.button("Stop", key="stop_query", use_container_width=True):
                st.session_state["is_generating"] = False
                st.info("No active generation to stop.")
        with send_col:
            st.button(
                "Submit",
                key="submit_query",
                type="primary",
                use_container_width=True,
                disabled=bool(st.session_state.get("is_generating")),
                on_click=queue_query_submission,
            )

    process_pending_query()


def run_client_app() -> None:
    st.set_page_config(page_title="VerifRAG", layout="wide")
    initialize_state()
    apply_styles()

    if st.query_params.get("vr_page") == PAGE_HOME:
        st.session_state["page"] = PAGE_HOME if st.session_state["authenticated"] else PAGE_LOGIN
        st.query_params.clear()

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
