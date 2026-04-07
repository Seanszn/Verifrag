from __future__ import annotations

import streamlit as st
from streamlit.testing.v1 import AppTest


def test_submit_button_calls_api_and_renders_response(monkeypatch):
    def fake_require_auth():
        return None

    def fake_render_header(*args, **kwargs):
        return None

    def fake_render_session_list(*args, **kwargs):
        return None

    def fake_render_chat_panel(messages):
        for message in messages:
            st.markdown(message["content"])

    def fake_ensure_active_session():
        if "sessions" not in st.session_state:
            st.session_state["sessions"] = [{"title": "Test", "messages": []}]
        if "active_session_index" not in st.session_state:
            st.session_state["active_session_index"] = 0
        return st.session_state["sessions"][st.session_state["active_session_index"]]

    def fake_submit_query():
        query = st.session_state["current_query"].strip() if "current_query" in st.session_state else ""
        session = fake_ensure_active_session()
        session["messages"].append({"role": "user", "content": query})
        session["messages"].append(
            {"role": "assistant", "content": "CourtListener test response"}
        )

    monkeypatch.setattr("src.client.ui.require_auth", fake_require_auth)
    monkeypatch.setattr("src.client.ui.render_header", fake_render_header)
    monkeypatch.setattr("src.client.ui.render_session_list", fake_render_session_list)
    monkeypatch.setattr("src.client.ui.render_chat_panel", fake_render_chat_panel)
    monkeypatch.setattr("src.client.ui.ensure_active_session", fake_ensure_active_session)
    monkeypatch.setattr("src.client.ui.submit_query", fake_submit_query)

    script = """
import streamlit as st
from src.client.ui import render_response_page

st.session_state["sessions"] = [{"title": "Test", "messages": []}]
st.session_state["active_session_index"] = 0
st.session_state["uploaded_files"] = []
st.session_state["upload_notice"] = None
st.session_state["clear_query_next_run"] = False
st.session_state["current_query"] = "negligence standard"
st.session_state["is_generating"] = False

render_response_page()
"""

    at = AppTest.from_string(script)
    at.run()

    assert len(at.text_area) == 1
    at.text_area[0].set_value("negligence standard")
    at.run()

    submit_buttons[0].click()
    at.run()   # handles the click and updates session state
    at.run()   # renders the updated messages on the page

    messages = at.session_state["sessions"][at.session_state["active_session_index"]]["messages"]
    assert any(m["content"] == "CourtListener test response" for m in messages)

    rendered = " ".join(
        [getattr(x, "value", "") for x in at.markdown]
        + [getattr(x, "value", "") for x in at.text]
        + [getattr(x, "value", "") for x in at.caption]
    )

    assert "CourtListener test response" in rendered