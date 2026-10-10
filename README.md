# EvidenceFlow AI

EvidenceFlow AI is an evidence-grounded document intelligence and RAG evaluation pipeline. Phase 1 establishes the repository foundation only. Ingestion and retrieval logic will be introduced in later verified phases.

## Phase 1 contents

- `src` package layout
- typed environment configuration
- structured JSON logging
- initial `Document` and `Chunk` domain records
- FastAPI health endpoint
- pytest configuration
- Ruff configuration
- optional dependency groups for RAG and UI components
- environment template

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Verify

```bash
python -m ruff check src tests
python -m pytest
python -m evidenceflow
```

## Start API

```bash
python -m uvicorn evidenceflow.api.app:app --reload
```

Health endpoint:

```text
http://127.0.0.1:8000/health
```

## Package layout

```text
src/evidenceflow/api       HTTP application
src/evidenceflow/core      configuration and logging
src/evidenceflow/domain    shared domain records
tests/unit                 isolated tests
tests/integration          cross-module/API tests
```

## Current status

Phase 1 foundation complete. No production retrieval or answer-generation claims are made at this stage.
