# HireLens AI

### RAG-Based Resume–Job Compatibility & Gap Analyzer

HireLens AI is an AI-powered application that compares a resume with a job
description using Retrieval-Augmented Generation (RAG), semantic embeddings,
ChromaDB, and Google Gemini.

It provides:

- Resume–JD match score
- Matched requirements
- Skill gaps
- Resume improvement suggestions

---

## Architecture

```text
Resume + Job Description
          │
          ▼
   Document Parser
          │
          ▼
    Resume Chunking
          │
          ▼
Sentence Transformers
   (Embeddings)
          │
          ▼
      ChromaDB
   (Semantic Search)
          │
          ▼
  JD Requirement Extraction
          │
          ▼
   Relevant Resume Evidence
          │
          ▼
     Google Gemini
          │
          ▼
 Match Score + Gaps + Suggestions
````

---

## Tech Stack

| Component           | Technology            |
| ------------------- | --------------------- |
| LLM                 | Google Gemini API     |
| Embeddings          | Sentence Transformers |
| Vector Database     | ChromaDB              |
| Backend             | FastAPI               |
| Frontend            | Streamlit             |
| Document Processing | PyPDF, python-docx    |
| Server              | Uvicorn               |
| Containerization    | Docker                |

---

## Project Structure

```text
HireLens-AI/
│
├── backend/
│   ├── main.py
│   ├── rag_engine.py
│   ├── document_parser.py
│   └── Dockerfile
│
├── frontend/
│   ├── app.py
│   └── Dockerfile
│
├── requirements.txt
├── docker-compose.yml
├── GEMINI_API_KEY.env
└── README.md
```

---

## Setup

### 1. Create Virtual Environment

Python 3.11 is recommended.

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure Gemini API Key

Create `GEMINI_API_KEY.env` in the project root:

```text
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

Do not commit this file to GitHub.

---

## Run Backend

Open **Terminal 1**:

```powershell
cd HireLens-AI
.\venv\Scripts\Activate.ps1

$key = (Get-Content .\GEMINI_API_KEY.env | Select-String '^GEMINI_API_KEY=').ToString().Split('=',2)[1].Trim()
$env:GEMINI_API_KEY = $key

cd backend
uvicorn main:app --reload --port 8000
```

Backend:

```text
http://localhost:8000
```

Swagger API documentation:

```text
http://localhost:8000/docs
```

---

## Run Frontend

Open **Terminal 2**:

```powershell
cd HireLens-AI
.\venv\Scripts\Activate.ps1

streamlit run .\frontend\app.py
```

Frontend:

```text
http://localhost:8501
```

Both backend and frontend must be running at the same time.

---

## How It Works

1. Parse the uploaded resume and job description.
2. Split the resume into text chunks.
3. Generate embeddings using Sentence Transformers.
4. Store embeddings in ChromaDB.
5. Extract job requirements using Gemini.
6. Retrieve relevant resume evidence using semantic search.
7. Send the retrieved evidence to Gemini.
8. Generate match score, gaps, and suggestions.

---

## API

### `GET /health`

Checks whether the backend is running.

```json
{
  "status": "ok"
}
```

### `POST /analyze`

Accepts:

* Resume file
* Job description text or file

Returns:

* Requirements analyzed
* Match analysis
* Matched requirements
* Gaps
* Suggestions

---

## Security

Add the following to `.gitignore`:

```text
venv/
__pycache__/
*.pyc
GEMINI_API_KEY.env
.env
```

Never commit your Gemini API key to GitHub.

---

## Future Improvements

* Multiple JD comparison
* Resume ranking
* Persistent vector database
* Authentication
* ATS analysis
* Resume rewriting
* Interview question generation
* Cloud deployment

---

## Author

**SriVidya Nagavalli**

B.Tech — Computer Science and Engineering
Shri Vishnu Engineering College for Women

```

This version is much more suitable for your GitHub: **short, technical, and enough to understand and run the project without making the README unnecessarily long.**
```
