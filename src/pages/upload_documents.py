import streamlit as st

st.title("Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDF or DOCX files",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"Uploaded {len(uploaded_files)} file(s).")

if st.button("Back to Home"):
    st.switch_page("app.py")