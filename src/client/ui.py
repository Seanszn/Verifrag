"""Spec-driven Streamlit client UI for the VeriRAG prototype."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import bcrypt
import base64
from src.storage.database import Database

import streamlit as st


PAGE_LOGIN = "login"
PAGE_HOME = "home"
PAGE_UPLOAD = "upload"
PAGE_RESPONSE = "response"
SUPPORTED_UPLOAD_TYPES = ["pdf", "txt", "docx", "md"]
TEMP_UPLOAD_DIR = Path(__file__).resolve().parents[2] / "temp_uploads"


@st.cache_resource
def get_db() -> Database:
    """Initialize and cache the database connection."""
    db = Database()
    db.initialize()
    return db

def initialize_state() -> None:
    """Initialize the session keys used by the prototype pages."""

    defaults: dict[str, Any] = {
        "authenticated": False,
        "page": PAGE_LOGIN,
        "uploaded_files": [],
        "sessions": [],
        "active_session_index": None,
        "current_query": "",
        "is_generating": False,
        "user": None,
        "show_register_notice": False,
        "show_upload_hint": False,
        "upload_notice": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

def apply_styles() -> None:
    """Inject light custom styling for the client-facing flow."""

    st.markdown(
        """
        <style>
        
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
        
        /* Ensure standard inputs are full width */
        div[data-testid="stTextInput"],
        div[data-testid="stTextInput"] > div {
            width: 100% !important;
        }

        /* NEW: Force Base Web password containers to full width */
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

        /* 1. Force Streamlit's main view container to use your gradient */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #f6f7f9 0%, #eef1f5 100%) !important;
        }

        /* 2. Turn the center content column into a floating white card to contrast with the gray background */
        .block-container {
            background: #ffffff;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
            border: 1rem Top, 2rem Right, 2rem Left, 1rem Bottom
            padding: 3rem 2rem !important;
            margin-top: 2rem;
            margin-bottom: 2rem;
            max-width: 1200px !important;
        }

        /* Target the horizontal block that contains the brand text */
        div[data-testid="columns"]:has(.vr-brand),
        div[data-testid="stHorizontalBlock"]:has(.vr-brand) {
            background: #121417;
            border: 1px solid #1f2937;
            border-radius: 20px;
            padding: 0.85rem 1rem;
            margin-top: -5rem !important; /* NEW: Force the bar to pull itself upwards */
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

        /* 1. Apply your custom border and background to the outer wrapper */
        div[data-baseweb="input"],
        div[data-baseweb="textarea"] {
            border-radius: 12px !important;
            border: 2px solid #8d8d8d !important;
            background: #ffffff !important;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        /* 2. Apply your custom blue focus state to the wrapper when clicked */
        div[data-baseweb="input"]:focus-within,
        div[data-baseweb="textarea"]:focus-within {
            border-color: #0f62fe !important;
            box-shadow: 0 0 0 1px #0f62fe !important;
        }

        /* 3. Strip the styling off the inner input tag so it blends perfectly into the new wrapper */
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
            background: #6b7280 !important; /* A nice, noticeable lighter grey */
            color: #ffffff !important;
            border-color: #6b7280 !important;
        }

        /* 1. Give the header button the same rounded pill shape as the others */
        div[data-testid="columns"]:has(.vr-brand) button,
        div[data-testid="stHorizontalBlock"]:has(.vr-brand) button {
            border-radius: 999px !important;
            font-weight: 600 !important;
        }

        /* 2. Target the hover state of the button inside the header */
        div[data-testid="columns"]:has(.vr-brand) button:hover,
        div[data-testid="stHorizontalBlock"]:has(.vr-brand) button:hover {
            background-color: #e5e7eb !important; 
            border-color: #e5e7eb !important;
            color: #111827 !important;
        }

        /* 3. Force the inner text elements to turn dark on hover */
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

        /* Hide the "Press Enter to apply" helper text */
        div[data-testid="InputInstructions"] {
            display: none !important;
            visibility: hidden !important;
        }

        /* Hide the default Streamlit header (with the Deploy button) */
        header[data-testid="stHeader"] {
            display: none !important;
            visibility: hidden !important;
        }
        
        /* Optional: Hide the default Streamlit footer "Made with Streamlit" */
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


def logout() -> None:
    """Clear local prototype state and return to login."""

    st.session_state["authenticated"] = False
    st.session_state["page"] = PAGE_LOGIN
    st.session_state["uploaded_files"] = []
    st.session_state["sessions"] = []
    st.session_state["active_session_index"] = None
    st.session_state["current_query"] = ""
    st.session_state["is_generating"] = False
    st.session_state["user"] = None
    st.session_state["show_register_notice"] = False
    st.session_state["show_upload_hint"] = False
    st.session_state["upload_notice"] = None
    st.rerun()


def render_header(action_label: str, action_callback) -> None:
    """Render the shared top header bar."""
    brand_col, action_col = st.columns([6, 1.5], vertical_alignment="center")
    with brand_col:
        st.markdown('<div class="vr-brand">VerifiRAG</div>', unsafe_allow_html=True)
    with action_col:
        # Use on_click so the callback runs BEFORE the script reruns
        st.button(
            action_label, 
            key=f"header_{st.session_state['page']}_{action_label}", 
            use_container_width=True,
            on_click=action_callback
        )


def require_auth() -> None:
    """Restrict authenticated pages to signed-in users."""

    if not st.session_state["authenticated"]:
        st.session_state["page"] = PAGE_LOGIN
        st.warning("Sign in to continue.")
        st.stop()


def render_page_intro(title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="vr-page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="vr-page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def format_session_title(timestamp: datetime | None = None) -> str:
    stamp = timestamp or datetime.now()
    return f"{stamp.month}/{stamp.day}/{str(stamp.year)[2:]}"


def build_placeholder_response(query: str, uploaded_files: list[dict[str, Any]]) -> str:
    """Return a client-demo response until the real RAG hook is wired in."""

    file_names = [item["name"] for item in uploaded_files]
    if file_names:
        source_line = f"Active uploaded sources: {', '.join(file_names)}."
    else:
        source_line = "No uploaded documents are attached to this session yet."

    return (
        f"Placeholder response for: {query}\n\n"
        f"{source_line}\n\n"
        "This page is wired for session history and can be replaced with the production RAG "
        "query function later."
    )


def process_uploaded_files(files: list[Any]) -> dict[str, Any]:
    """Persist uploaded files to a local temp directory and return metadata."""

    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_files: list[dict[str, Any]] = []

    for uploaded in files:
        filename = Path(uploaded.name).name
        destination = TEMP_UPLOAD_DIR / filename
        destination.write_bytes(bytes(uploaded.getbuffer()))
        saved_files.append(
            {
                "name": filename,
                "size": int(uploaded.size),
                "path": str(destination),
            }
        )

    return {
        "success": bool(saved_files),
        "file_count": len(saved_files),
        "filenames": [item["name"] for item in saved_files],
        "files": saved_files,
    }


def render_session_list(sessions: list[dict[str, Any]], active_session_index: int | None) -> None:
    """Render the left-side list of prior sessions."""

    with st.container(border=True):
        st.markdown('<div class="vr-panel-label">Previous Sessions:</div>', unsafe_allow_html=True)

        if not sessions:
            st.markdown('<div class="vr-muted">No prior sessions yet.</div>', unsafe_allow_html=True)
            return

        st.markdown('<div class="vr-session-anchor"></div>', unsafe_allow_html=True)
        for index, session in enumerate(sessions):
            label = session["title"]
            if active_session_index == index:
                label = f"{label} - Active"
            if st.button(label, key=f"session_{index}", use_container_width=True):
                st.session_state["active_session_index"] = index
                st.rerun()


def render_chat_panel(messages: list[dict[str, str]]) -> None:
    """Render the transcript for the active session."""

    with st.container(border=True, height=430):
        if not messages:
            st.markdown(
                '<div class="vr-muted">This is where generated responses will appear.</div>',
                unsafe_allow_html=True,
            )
            return

        for message in messages:
            role = "assistant" if message["role"] == "assistant" else "user"
            with st.chat_message(role):
                st.markdown(message["content"])


def render_upload_panel() -> list[Any]:
    """Render the centered upload card and return current widget files."""

    with st.container(border=True):
        st.markdown('<div class="vr-panel-label">Upload Zone</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="vr-muted">Drag and drop files here, or click Browse files inside the box.</div>',
            unsafe_allow_html=True,
        )
        return st.file_uploader(
            "Upload source documents",
            type=SUPPORTED_UPLOAD_TYPES,
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
    

def render_login_page() -> None:
    # Toggle between Login and Register modes
    mode_label = "Switch to Login" if st.session_state.get("show_register_notice") else "Register"
    
    def toggle_mode():
        st.session_state["show_register_notice"] = not st.session_state.get("show_register_notice", False)
        
    render_header(mode_label, toggle_mode)

    is_registering = st.session_state.get("show_register_notice", False)
    db = get_db()

    _, form_col, _ = st.columns([1.2, 1.6, 1.2])
    with form_col:
        title_text = "Create an " if is_registering else "Sign into your "
        st.markdown(
            f'<div class="vr-page-title"><span class="vr-page-title-muted">{title_text}</span>account</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="vr-login-anchor"></div>', unsafe_allow_html=True)
        username = st.text_input("Username", key="login_username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
        
        st.write("")
        st.write("")
        # Note: We removed the 2FA Passkey input to match the database schema

        button_label = "Register Account" if is_registering else "Sign In"
        
        if st.button(button_label, key="login_submit", type="primary", use_container_width=True):
            username_val = username.strip()
            
            if not username_val or not password:
                st.error("Please fill out both fields.")
                return

            # --- REGISTRATION FLOW ---
            if is_registering:
                # Check if user already exists using Sean's method
                if db.get_user_by_username(username_val):
                    st.error("Username already exists. Please choose another.")
                    return
                
                # Hash the password and save to DB
                salt = bcrypt.gensalt()
                hashed_pw = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
                
                try:
                    db.create_user(username_val, hashed_pw)
                    st.success("Registration successful! Switching to login...")
                    st.session_state["show_register_notice"] = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Database error: {str(e)}")

            # --- LOGIN FLOW ---
            else:
                user_row = db.get_user_by_username(username_val)
                
                if user_row:
                    # Verify the hash
                    stored_hash = user_row["password_hash"]
                    if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
                        st.session_state["authenticated"] = True
                        
                        # Store the database User ID in session state for later use
                        st.session_state["user"] = {
                            "id": user_row["id"], 
                            "username": user_row["username"]
                        }
                        st.session_state["page"] = PAGE_HOME
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                else:
                    st.error("Invalid username or password.")
                    
'''
def render_login_page() -> None:
    render_header("Register", lambda: st.session_state.__setitem__("show_register_notice", True))

    _, form_col, _ = st.columns([1.2, 1.6, 1.2])
    with form_col:
        st.markdown(
            '<div class="vr-page-title"><span class="vr-page-title-muted">Sign into your </span>account</div>',
            unsafe_allow_html=True,
        )
        if st.session_state["show_register_notice"]:
            st.info("Registration is not yet implemented in this prototype.")

        st.markdown('<div class="vr-login-anchor"></div>', unsafe_allow_html=True)
        username = st.text_input("Username", key="login_username", placeholder="Enter your username")
        password = st.text_input(
            "Password",
            type="password",
            key="login_password",
            placeholder="Enter your password",
        )
        passkey = st.text_input("2FA Passkey", key="login_passkey", placeholder="Enter your passkey")

        if st.button("Submit", key="login_submit", type="primary", use_container_width=True):
            missing_fields = [
                label
                for label, value in [
                    ("Username", username),
                    ("Password", password),
                    ("2FA Passkey", passkey),
                ]
                if not str(value).strip()
            ]
            if missing_fields:
                st.error(f"Enter: {', '.join(missing_fields)}.")
            else:
                st.session_state["authenticated"] = True
                st.session_state["user"] = {"username": username.strip()}
                st.session_state["page"] = PAGE_HOME
                st.session_state["show_register_notice"] = False
                st.rerun()
'''

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
            navigate(PAGE_RESPONSE)
        st.write("")
        if st.button("Upload Documents", key="home_upload", use_container_width=True):
            navigate(PAGE_UPLOAD)

def render_upload_page() -> None:
    # 1. Standard Page Setup (Brings back the Black Bar and Title)
    require_auth()
    render_header("Home", lambda: navigate(PAGE_HOME))
    render_page_intro("Test Upload Query Page", "Add source documents before asking questions.")

    # 2. Icon Logic
    icon_base64 = ""
    icon_path = Path(__file__).parent.parent.parent / "assets" / "upload.png"
    if icon_path.exists():
        with open(icon_path, "rb") as f:
            icon_base64 = base64.b64encode(f.read()).decode()

    # 3. Your Custom Uploader CSS
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

        /* Custom styling for the instruction text */
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

    # 4. Layout & Logic
    _, center, _ = st.columns([0.7, 2.2, 0.7])
    with center:
        uploaded_files = st.file_uploader(
            "",
            type=SUPPORTED_UPLOAD_TYPES,
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        # NEW: Instruction text added here
        st.markdown(
            '<p class="upload-instruction">Use the upload box above to drag files in or click to browse files</p>', 
            unsafe_allow_html=True
        )

        if uploaded_files:
            st.write("")
            st.markdown("**Selected Files:**")
            for file in uploaded_files:
                st.markdown(f"📄 {file.name}")

        st.write("")
        # Centering the Submit button
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            submit_clicked = st.button("Submit", key="submit_upload", type="primary", use_container_width=True)

        if submit_clicked:
            if not uploaded_files:
                st.error("Select at least one file before submitting.")
            else:
                upload_result = process_uploaded_files(list(uploaded_files))
                st.session_state["uploaded_files"] = upload_result["files"]
                st.session_state["upload_notice"] = f"Uploaded {upload_result['file_count']} file(s)."
                navigate(PAGE_RESPONSE)

'''

def render_upload_page() -> None:
    # 1. Standard Page Setup (Brings back the Black Bar and Title)
    require_auth()
    render_header("Home", lambda: navigate(PAGE_HOME))
    render_page_intro("Test Upload Query Page", "Add source documents before asking questions.")

    # 2. Icon Logic
    icon_base64 = ""
    icon_path = Path(__file__).parent.parent.parent / "assets" / "upload.png"
    if icon_path.exists():
        with open(icon_path, "rb") as f:
            icon_base64 = base64.b64encode(f.read()).decode()

    # 3. Your Custom Uploader CSS
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
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 4. Layout & Logic (Brings back the Submit Button)
    _, center, _ = st.columns([0.7, 2.2, 0.7])
    with center:
        uploaded_files = st.file_uploader(
            "",
            type=SUPPORTED_UPLOAD_TYPES,
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            st.write("")
            st.markdown("**Selected Files:**")
            for file in uploaded_files:
                st.markdown(f"📄 {file.name}")

        st.write("")
        # Centering the Submit button
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            submit_clicked = st.button("Submit", key="submit_upload", type="primary", use_container_width=True)

        if submit_clicked:
            if not uploaded_files:
                st.error("Select at least one file before submitting.")
            else:
                upload_result = process_uploaded_files(list(uploaded_files))
                st.session_state["uploaded_files"] = upload_result["files"]
                st.session_state["upload_notice"] = f"Uploaded {upload_result['file_count']} file(s)."
                navigate(PAGE_RESPONSE)
'''

def ensure_active_session() -> dict[str, Any]:
    """Return the active session, creating one only when needed."""

    sessions = st.session_state["sessions"]
    active_index = st.session_state["active_session_index"]

    if active_index is None or active_index >= len(sessions):
        return {"title": format_session_title(), "messages": []}
    return sessions[active_index]


def save_active_session(session: dict[str, Any]) -> None:
    """Persist the current session into session state."""

    sessions = st.session_state["sessions"]
    active_index = st.session_state["active_session_index"]

    if active_index is None or active_index >= len(sessions):
        sessions.insert(0, session)
        st.session_state["active_session_index"] = 0
        return

    sessions[active_index] = session


def submit_query() -> None:
    """Append the user query and placeholder assistant response."""

    query = st.session_state["current_query"].strip()
    if not query:
        st.error("Enter a query before submitting.")
        return

    st.session_state["is_generating"] = True
    session = ensure_active_session()
    session["messages"].append({"role": "user", "content": query})
    response = build_placeholder_response(query, st.session_state["uploaded_files"])
    session["messages"].append({"role": "assistant", "content": response})
    save_active_session(session)
    st.session_state["current_query"] = ""
    st.session_state["is_generating"] = False
    st.rerun()


def render_response_page() -> None:
    require_auth()
    render_header("Home", lambda: navigate(PAGE_HOME))

    if st.session_state.get("upload_notice"):
        st.success(st.session_state["upload_notice"])
        st.session_state["upload_notice"] = None

    left_col, right_col = st.columns([1.1, 2.4], gap="large")

    with left_col:
        render_session_list(
            st.session_state["sessions"],
            st.session_state["active_session_index"],
        )

        if st.session_state["uploaded_files"]:
            st.write("")
            with st.container(border=True):
                st.markdown('<div class="vr-panel-label">Uploaded Sources</div>', unsafe_allow_html=True)
                for file in st.session_state["uploaded_files"]:
                    st.markdown(f"- {file['name']} ({file['size']} bytes)")

    with right_col:
        active_session = ensure_active_session()
        render_chat_panel(active_session["messages"])
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
                st.info("No active generation to stop in this prototype yet.")
        with send_col:
            if st.button("Submit", key="submit_query", type="primary", use_container_width=True):
                submit_query()


def run_client_app() -> None:
    """Run the Streamlit GUI prototype."""

    st.set_page_config(page_title="VerifRAG", layout="wide")
    initialize_state()
    apply_styles()

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
