# Architecture Decisions

## ADR-001: Use a `src` package layout

Status: Accepted

Reason: Prevent accidental imports from the repository root and make editable-install behavior match packaged behavior.

## ADR-002: Use standard-library JSON logging

Status: Accepted

Reason: Establish structured logs without adding an unnecessary runtime dependency.

## ADR-003: Keep heavy AI dependencies optional in Phase 1

Status: Accepted

Reason: Foundation tests and the health endpoint should run without downloading models. RAG dependencies are installed through the `rag` optional dependency group when needed.

## ADR-004: Disable the reranker by default

Status: Accepted

Reason: Phase 0 requires measured retrieval improvement and acceptable CPU latency before the reranker becomes a default stage.

## ADR-005: Use immutable domain records where practical

Status: Accepted

Reason: Documents and chunks should be safely passed between pipeline stages without hidden mutation.
