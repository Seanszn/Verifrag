import streamlit as st

st.title("Search")

query = st.text_input("Enter your legal question")

if st.button("Run Search"):
    st.write(f"Searching for: {query}")
    st.session_state["last_query"] = query

if st.button("Go to Verification"):
    st.switch_page("pages/3_Verification.py")

if st.button("Back to Home"):
    st.switch_page("app.py")