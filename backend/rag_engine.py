"""
rag_engine.py
-------------
Core RAG logic:
  1. Chunk resume + JD text
  2. Embed chunks using Sentence Transformers
  3. Store resume chunks in ChromaDB
  4. Retrieve relevant resume evidence for each JD requirement
  5. Use Google Gemini REST API for requirement extraction and analysis
"""

import os
import uuid
import re
import time
from typing import List, Dict

import chromadb
import requests
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_embedder = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

_chroma_client = chromadb.EphemeralClient()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Primary model
LLM_MODEL = "gemini-3.5-flash-lite"

# Fallback models
FALLBACK_MODELS = [
    "gemini-3.5-flash",
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
]

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )


# ---------------------------------------------------------------------------
# Gemini REST API
# ---------------------------------------------------------------------------

def call_llm(
    prompt: str,
    max_tokens: int = 1000
) -> str:
    """
    Call Gemini through the REST API.

    The API key is sent using x-goog-api-key.

    If the primary model is temporarily unavailable (503),
    automatically tries fallback models.
    """

    models = [
        LLM_MODEL,
        *[
            model
            for model in FALLBACK_MODELS
            if model != LLM_MODEL
        ]
    ]

    last_error = None

    for model in models:

        url = (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{model}:generateContent"
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": max_tokens
            }
        }

        try:

            response = requests.post(
                url,
                headers={
                    "x-goog-api-key": GEMINI_API_KEY,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )

            # Successful response
            if response.ok:

                data = response.json()

                try:
                    return (
                        data["candidates"][0]
                        ["content"]["parts"][0]["text"]
                    )

                except (
                    KeyError,
                    IndexError,
                    TypeError
                ):
                    raise RuntimeError(
                        f"Unexpected Gemini API response: {data}"
                    )

            # Temporary overload
            if response.status_code == 503:

                last_error = (
                    f"Gemini model {model} returned 503: "
                    f"{response.text}"
                )

                # Try the next model
                continue

            # Authentication
            if response.status_code == 401:

                raise RuntimeError(
                    "Gemini API authentication failed. "
                    "Check GEMINI_API_KEY."
                )

            # Other API errors
            raise RuntimeError(
                f"Gemini API error {response.status_code}: "
                f"{response.text}"
            )

        except requests.RequestException as e:

            last_error = (
                f"Network error while calling Gemini "
                f"model {model}: {str(e)}"
            )

            continue

    # All models failed
    raise RuntimeError(
        f"All Gemini models failed. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50
) -> List[str]:
    """
    Split text into overlapping chunks by word count.
    """

    words = text.split()

    if not words:
        return []

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ---------------------------------------------------------------------------
# Vector store operations
# ---------------------------------------------------------------------------

def build_collection(
    resume_chunks: List[str]
) -> tuple:
    """
    Create a fresh Chroma collection for this request.
    """

    collection_name = (
        f"resume_{uuid.uuid4().hex[:8]}"
    )

    collection = (
        _chroma_client.create_collection(
            name=collection_name
        )
    )

    if not resume_chunks:
        return (
            collection,
            collection_name
        )

    embeddings = (
        _embedder
        .encode(resume_chunks)
        .tolist()
    )

    ids = [
        f"chunk_{i}"
        for i in range(len(resume_chunks))
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=resume_chunks,
    )

    return (
        collection,
        collection_name
    )


def retrieve_relevant_chunks(
    collection,
    query: str,
    top_k: int = 3
) -> List[str]:
    """
    Retrieve the most relevant resume chunks.
    """

    query_embedding = (
        _embedder
        .encode([query])
        .tolist()
    )

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    return (
        results["documents"][0]
        if results["documents"]
        else []
    )


def cleanup_collection(
    collection_name: str
):
    """
    Delete the temporary Chroma collection.
    """

    try:

        _chroma_client.delete_collection(
            name=collection_name
        )

    except Exception:
        pass


# ---------------------------------------------------------------------------
# JD requirement extraction
# ---------------------------------------------------------------------------

def extract_jd_requirements(
    jd_text: str
) -> List[str]:
    """
    Ask Gemini to extract concrete JD requirements.
    """

    prompt = f"""You are analyzing a job description.

Extract a clean, deduplicated list of the concrete requirements:

- required technical skills
- programming languages
- frameworks
- databases
- tools
- qualifications
- important responsibilities

One requirement per line.
Do not number the requirements.
Do not add explanations.

Job Description:
{jd_text}

Return ONLY the list, one requirement per line."""

    raw = call_llm(
        prompt,
        max_tokens=600
    )

    requirements = []

    for line in raw.splitlines():

        if not line.strip():
            continue

        cleaned = re.sub(
            r"^[-*•\d.\s]+",
            "",
            line
        ).strip()

        if cleaned:
            requirements.append(cleaned)

    return requirements


# ---------------------------------------------------------------------------
# Match analysis
# ---------------------------------------------------------------------------

def analyze_match(
    resume_text: str,
    jd_text: str
) -> Dict:
    """
    Complete RAG pipeline:

    Resume
        ↓
    Chunking
        ↓
    Sentence Transformer embeddings
        ↓
    ChromaDB
        ↓
    JD requirement extraction
        ↓
    Semantic retrieval
        ↓
    Gemini analysis
        ↓
    Match score + matched skills + gaps + suggestions
    """

    # -------------------------------------------------------
    # 1. Chunk resume
    # -------------------------------------------------------

    resume_chunks = chunk_text(
        resume_text
    )

    # -------------------------------------------------------
    # 2. Build vector database
    # -------------------------------------------------------

    collection, collection_name = (
        build_collection(resume_chunks)
    )

    try:

        # ---------------------------------------------------
        # 3. Extract JD requirements
        # ---------------------------------------------------

        requirements = (
            extract_jd_requirements(
                jd_text
            )
        )

        # ---------------------------------------------------
        # 4. Retrieve resume evidence
        # ---------------------------------------------------

        evidence_map = {}

        for requirement in requirements:

            relevant = (
                retrieve_relevant_chunks(
                    collection,
                    requirement,
                    top_k=2
                )
            )

            evidence_map[
                requirement
            ] = relevant

    finally:

        # Always clean Chroma collection
        cleanup_collection(
            collection_name
        )

    # -------------------------------------------------------
    # 5. Build evidence block
    # -------------------------------------------------------

    evidence_block = "\n\n".join(

        f"Requirement: {requirement}\n"
        f"Resume evidence found: "
        f"{' | '.join(chunks) if chunks else 'NONE FOUND'}"

        for requirement, chunks
        in evidence_map.items()
    )

    # -------------------------------------------------------
    # 6. Final recruiter analysis
    # -------------------------------------------------------

    final_prompt = f"""You are an expert technical recruiter.

Analyze the candidate's resume against the job requirements.

Use ONLY the evidence retrieved from the candidate's resume.

Produce:

1. Overall match score out of 100.

2. MATCHED REQUIREMENTS
   - List the requirements clearly.
   - Give a one-line justification for each.

3. GAPS
   - List missing or weak requirements.
   - Give a one-line explanation for each.

4. SUGGESTIONS
   - Give 3-5 specific and actionable suggestions.
   - Focus on improving the resume for this particular job.
   - Suggest rewording existing experience where appropriate.
   - Suggest highlighting relevant skills/projects.
   - Do not invent experience the candidate does not have.

Evidence retrieved from resume:

{evidence_block}

Respond using exactly these Markdown headers:

## Match Score

## Matched Requirements

## Gaps

## Suggestions
"""

    analysis_text = call_llm(
        final_prompt,
        max_tokens=1200
    )

    return {
        "requirements_analyzed": len(
            requirements
        ),
        "analysis_markdown": analysis_text,
    }