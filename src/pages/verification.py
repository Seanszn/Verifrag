import streamlit as st

st.title("Verification Results")

query = st.session_state.get("last_query", "")

if query:
    st.write(f"Showing verification results for: {query}")
else:
    st.info("No query has been run yet.")

if st.button("Back to Search"):
    st.switch_page("pages/2_Search.py")

if st.button("Back to Home"):
    st.switch_page("app.py")