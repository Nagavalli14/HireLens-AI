"""
app.py
------
Streamlit frontend for the Resume-JD Matcher.
Lets the user upload a resume and paste/upload a JD, then calls the
FastAPI backend and displays the match analysis.
"""

import os
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Resume-JD Matcher", page_icon="📄", layout="centered")

st.title("📄 Resume ↔ Job Description Matcher")
st.caption(
    "Upload your resume and a job description. This tool uses a RAG pipeline "
    "(semantic retrieval + LLM analysis) to score your fit and suggest improvements."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Your Resume")
    resume_file = st.file_uploader(
        "Upload resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"]
    )

with col2:
    st.subheader("2. Job Description")
    jd_input_mode = st.radio("How do you want to provide the JD?", ["Paste text", "Upload file"])
    jd_text = None
    jd_file = None
    if jd_input_mode == "Paste text":
        jd_text = st.text_area("Paste the job description here", height=200)
    else:
        jd_file = st.file_uploader(
            "Upload JD (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"], key="jd_upload"
        )

st.divider()

if st.button("🔍 Analyze Match", type="primary", use_container_width=True):
    if not resume_file:
        st.error("Please upload your resume.")
    elif jd_input_mode == "Paste text" and not jd_text:
        st.error("Please paste a job description.")
    elif jd_input_mode == "Upload file" and not jd_file:
        st.error("Please upload a job description file.")
    else:
        with st.spinner("Running retrieval + analysis... this takes ~10-20 seconds"):
            try:
                files = {
                    "resume_file": (resume_file.name, resume_file.getvalue()),
                }
                data = {}
                if jd_text:
                    data["jd_text"] = jd_text
                if jd_file:
                    files["jd_file"] = (jd_file.name, jd_file.getvalue())

                response = requests.post(
                    f"{BACKEND_URL}/analyze", files=files, data=data, timeout=120
                )

                if response.status_code == 200:
                    result = response.json()
                    st.success(
                        f"Analysis complete — {result['requirements_analyzed']} "
                        f"requirements extracted from the JD."
                    )
                    st.markdown(result["analysis_markdown"])
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

            except requests.exceptions.ConnectionError:
                st.error(
                    f"Could not reach the backend at {BACKEND_URL}. "
                    "Make sure the FastAPI server is running."
                )
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

st.divider()
st.caption("Built with LangChain-style RAG concepts, ChromaDB, sentence-transformers, and Claude.")
