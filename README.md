# EvidenceFlow AI

**A production-oriented, evidence-grounded document RAG chatbot with evaluation and observability.**

EvidenceFlow AI lets users upload a PDF, Markdown, or text document and ask questions through a conversational interface. It combines sparse and dense retrieval, rank fusion, cross-encoder reranking, LLM-based grounded generation, citation validation, intelligent refusal handling, and retrieval diagnostics.

## Key Features

- Document ingestion for PDF, Markdown, and plain-text files
- Structure-aware text chunking with configurable overlap
- Persistent document, chunk, and embedding storage in SQLite
- BM25 sparse retrieval for lexical matches
- Sentence Transformer embeddings for semantic retrieval
- Weighted Reciprocal Rank Fusion for hybrid search
- Cross-encoder reranking for stronger evidence selection
- Document-scoped retrieval to prevent cross-document leakage
- Groq-powered grounded answer generation
- Optional secondary LLM provider and extractive fail-safe fallback
- Conversation history for contextual follow-up questions
- Exact chunk citations attached to factual statements and bullets
- Citation ID, coverage, and evidence validation
- Explicit refusal when the active document cannot support an answer
- Smart evidence sufficiency using sparse and dense retrieval signals
- FastAPI backend, Streamlit chatbot, and command-line workflows
- Structured logging, latency tracking, diagnostics, and evaluation outputs
- Automated quality checks with Pytest and Ruff

## System Architecture

```mermaid
flowchart LR
    A[Upload PDF, Markdown, or TXT] --> B[Document Ingestion]
    B --> C[Structure-Aware Chunking]
    C --> D[Sentence Transformer Embeddings]
    C --> E[BM25 Sparse Index]
    D --> F[Dense Semantic Search]
    E --> G[Weighted RRF Fusion]
    F --> G
    G --> H[Cross-Encoder Reranking]
    H --> I[Evidence Sufficiency Check]
    I -->|Supported| J[Grounded LLM Generation]
    I -->|Unsupported| K[Explicit Refusal]
    J --> L[Citation Validation]
    L --> M[Document-Scoped Chat Response]
    M --> N[Sources, Evidence, and Diagnostics]
```

## Retrieval and Answering Flow

1. A document is ingested and assigned a unique document ID.
2. The document is divided into structured, overlapping chunks.
3. Chunk embeddings are generated and persisted.
4. A user question is searched through BM25 and dense semantic retrieval.
5. Sparse and dense rankings are combined using weighted Reciprocal Rank Fusion.
6. A cross-encoder reranks the strongest candidates.
7. Evidence sufficiency is evaluated using retrieval signals and an optional advanced reranker threshold.
8. The LLM receives only the selected evidence and recent conversation context.
9. Every factual sentence or bullet must contain an exact chunk citation.
10. Citation IDs are validated against the retrieved evidence before the answer is returned.
11. Unsupported questions receive an explicit refusal instead of an ungrounded answer.

## Technology Stack

- **Language:** Python 3.12
- **API:** FastAPI, Pydantic, Uvicorn
- **User Interface:** Streamlit
- **Retrieval:** BM25, Sentence Transformers, weighted RRF
- **Reranking:** Cross-encoder reranker
- **LLM:** Groq, with optional secondary provider support
- **Storage:** SQLite, NumPy vector serialization
- **Document Processing:** PyPDF
- **Testing and Quality:** Pytest, Ruff
- **Configuration:** python-dotenv

## Project Structure

```text
evidenceflow-ai/
├── data/
│   ├── documents/
│   └── evidenceflow.db
├── src/evidenceflow/
│   ├── api/             # FastAPI app, routes, schemas, dependencies
│   ├── chunking/        # Structure-aware chunk creation
│   ├── core/            # Configuration and logging
│   ├── domain/          # Shared domain models
│   ├── embeddings/      # Sentence Transformer encoding
│   ├── evaluation/      # Retrieval and answer evaluation
│   ├── generation/      # Sufficiency checks and citation validation
│   ├── indexing/        # Embedding indexing and semantic search
│   ├── ingestion/       # PDF, Markdown, and text ingestion
│   ├── llm/             # Providers, prompts, fallback orchestration
│   ├── reranking/       # Cross-encoder reranking
│   ├── retrieval/       # BM25, hybrid search, and RRF
│   ├── storage/         # SQLite repositories
│   └── ui/              # Streamlit chatbot and API client
├── tests/
│   ├── integration/
│   └── unit/
├── .env.example
├── Makefile
├── pyproject.toml
└── README.md
```

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/NiranjanGhising/EvidenceFlow-AI.git
cd EvidenceFlow-AI
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install the project

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,rag,ui]"
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key
EVIDENCEFLOW_GROQ_MODEL=openai/gpt-oss-20b

# Optional secondary provider
GEMINI_API_KEY=
EVIDENCEFLOW_GEMINI_MODEL=gemini-3.8-flash
```

Do not commit `.env` or any API key.

### 5. Run quality checks

```bash
python -m ruff check src tests
python -m pytest -q
```

### 6. Start the API

```bash
make api
```

The API is available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### 7. Start the Streamlit interface

Open a second terminal, activate the virtual environment, and run:

```bash
make ui
```

Open:

```text
http://localhost:8501
```

## How to Use

1. Open the Streamlit interface.
2. Upload a PDF, Markdown, or text document.
3. Configure chunk size and overlap if required.
4. Process the document to ingest, chunk, and index it.
5. Open the document-chat section.
6. Ask a question about the active document.
7. Continue with contextual follow-up questions.
8. Expand **Sources**, **Retrieved evidence**, or **Technical details** for observability.
9. Use **Clear chat** when starting a new conversation.
10. Uploading and processing another document makes it the active document for future questions.

## Example Questions

```text
What projects are listed in this document?
```

```text
Which of those projects uses Python?
```

```text
What data engineering skills are described?
```

If the active document cannot support an answer, EvidenceFlow returns an explicit refusal.

## Command-Line Answering

After a document has been processed and indexed:

```bash
evidenceflow-answer \
  "What does the document say about FastAPI?" \
  --candidates 10 \
  --limit 5
```

## API Examples

### Document-scoped hybrid search

```bash
curl -X POST http://127.0.0.1:8000/api/v1/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What projects use Python?",
    "document_id": "doc-example",
    "candidates": 10,
    "limit": 5
  }'
```

### Grounded conversational answer

```bash
curl -X POST http://127.0.0.1:8000/api/v1/answers \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Which of those projects uses Python?",
    "document_id": "doc-example",
    "conversation_history": [
      {
        "role": "user",
        "content": "What projects are listed?"
      },
      {
        "role": "assistant",
        "content": "The document lists several projects."
      }
    ],
    "candidates": 10,
    "limit": 5,
    "minimum_rerank_score": null
  }'
```

## Testing

Run the complete suite:

```bash
python -m pytest -q
```

Run linting:

```bash
python -m ruff check src tests
```

Apply safe Ruff fixes:

```bash
python -m ruff check src tests --fix
```

Compile-check the source package:

```bash
python -m compileall -q src/evidenceflow
```

The current implementation passes **69 automated tests**.

## Reliability and Safety Design

EvidenceFlow is intentionally stricter than a general-purpose chatbot:

- Answers are restricted to the active document.
- Conversation history helps interpret references but is not treated as factual evidence.
- Every factual statement must include a valid chunk citation.
- Citation IDs must exist in the selected evidence.
- Unsupported questions are refused.
- A provider failure can fall back to another configured provider or an extractive answer.
- Retrieval, reranking, generation, citation validation, fallbacks, and latency are observable.

## Current Limitations

- Exact vector search is stored in SQLite and is intended for local or moderate-scale use.
- The first model load may require downloading Sentence Transformer and cross-encoder weights.
- Career advice, recommendations, and external knowledge are refused unless explicitly supported by the active document.
- A production deployment should add authentication, rate limiting, secrets management, background processing, and a scalable vector database.

## Roadmap

- Dockerized deployment
- Authentication and multi-user document isolation
- Asynchronous document processing
- Streaming answer generation
- Larger-scale vector database support
- Retrieval evaluation datasets and automated regression reports
- Cloud deployment and monitoring

## GitHub Push Guide

### First push to a new repository

Create an empty repository named `EvidenceFlow-AI` on GitHub. Do not add a README, license, or `.gitignore during repository creation if those files already exist locally.

From the project root:

```bash
git status
git add .
git commit -m "feat: complete document-scoped RAG chatbot"
git branch -M main
git remote add origin https://github.com/NiranjanGhising/EvidenceFlow-AI.git
git push -u origin main
```

If `origin` already exists:

```bash
git remote -v
git remote set-url origin https://github.com/NiranjanGhising/EvidenceFlow-AI.git
git push -u origin main
```

### Push future updates

```bash
git status
git add .
git commit -m "docs: improve project documentation"
git push origin main
```

### Safety check before pushing

```bash
git status
git diff --cached
git ls-files | grep -E '(^|/)\.env$|\.pem$|\.key$'
```

The final command should return no secret files.

Ensure `.gitignore` contains at least:

```gitignore
.env
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/
data/*.db
data/documents/*
!data/documents/.gitkeep
```

If `.env` was accidentally staged:

```bash
git rm --cached .env
echo ".env" >> .gitignore
git add .gitignore
git commit -m "chore: remove environment secrets from tracking"
```

If an API key was ever committed, revoke and replace the key before pushing again. Removing the file from the latest commit does not remove the secret from Git history.

## Author

**Niranjan Ghising**

- GitHub: https://github.com/NiranjanGhising
- LinkedIn: https://www.linkedin.com/in/niranjan-ghising-04500424a/

---

Built to demonstrate reliable RAG engineering through grounded generation, document isolation, citation validation, conversational context, evaluation, and observability.
