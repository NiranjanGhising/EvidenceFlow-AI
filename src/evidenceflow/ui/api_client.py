"""HTTP client for communicating with the EvidenceFlow API."""

from __future__ import annotations

from typing import Any

import httpx


class EvidenceFlowAPIError(RuntimeError):
    """Raised when the EvidenceFlow API request fails."""


class EvidenceFlowAPIClient:
    """Client for EvidenceFlow API endpoints."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout_seconds: float = 120.0,
    ) -> None:
        normalized_url = base_url.strip().rstrip("/")

        if not normalized_url:
            raise ValueError("base_url cannot be empty")

        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than zero"
            )

        self.base_url = normalized_url
        self.timeout_seconds = timeout_seconds

    def _request(
        self,
        method: str,
        endpoint: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send an API request and return its JSON response."""

        url = f"{self.base_url}{endpoint}"

        try:
            with httpx.Client(
                timeout=self.timeout_seconds
            ) as client:
                response = client.request(
                    method=method,
                    url=url,
                    json=payload,
                )
        except httpx.ConnectError as exc:
            raise EvidenceFlowAPIError(
                "Unable to connect to the EvidenceFlow API. "
                "Start the API with 'make api'."
            ) from exc
        except httpx.TimeoutException as exc:
            raise EvidenceFlowAPIError(
                "The EvidenceFlow API request timed out."
            ) from exc
        except httpx.HTTPError as exc:
            raise EvidenceFlowAPIError(
                f"EvidenceFlow API request failed: {exc}"
            ) from exc

        if response.is_error:
            try:
                error_data = response.json()
            except ValueError:
                error_data = {
                    "detail": response.text
                }

            detail = error_data.get(
                "detail",
                error_data,
            )

            if isinstance(detail, dict):
                message = str(
                    detail.get(
                        "detail",
                        detail.get(
                            "error",
                            "API request failed",
                        ),
                    )
                )
            else:
                message = str(detail)

            raise EvidenceFlowAPIError(
                f"API returned HTTP "
                f"{response.status_code}: {message}"
            )

        try:
            response_data = response.json()
        except ValueError as exc:
            raise EvidenceFlowAPIError(
                "The EvidenceFlow API returned invalid JSON."
            ) from exc

        if not isinstance(response_data, dict):
            raise EvidenceFlowAPIError(
                "The EvidenceFlow API returned an "
                "unexpected response format."
            )

        return response_data

    def health(self) -> dict[str, Any]:
        """Return API health information."""

        return self._request(
            method="GET",
            endpoint="/health",
        )

    def ingest_document(
        self,
        path: str,
    ) -> dict[str, Any]:
        """Ingest a PDF, Markdown, or text document."""

        return self._request(
            method="POST",
            endpoint="/api/v1/documents/ingest",
            payload={
                "path": path,
            },
        )

    def chunk_document(
        self,
        document_id: str,
        chunk_size: int = 800,
        overlap: int = 120,
    ) -> dict[str, Any]:
        """Chunk a previously ingested document."""

        return self._request(
            method="POST",
            endpoint="/api/v1/documents/chunk",
            payload={
                "document_id": document_id,
                "chunk_size": chunk_size,
                "overlap": overlap,
            },
        )

    def index_document(
        self,
        document_id: str,
    ) -> dict[str, Any]:
        """Generate and persist document embeddings."""

        return self._request(
            method="POST",
            endpoint="/api/v1/documents/index",
            payload={
                "document_id": document_id,
            },
        )

    def hybrid_search(
        self,
        query: str,
        limit: int = 5,
        candidates: int = 20,
        document_id: str | None = None,
    ) -> dict[str, Any]:
        """Run document-scoped hybrid retrieval."""

        return self._request(
            method="POST",
            endpoint="/api/v1/search/hybrid",
            payload={
                "query": query,
                "document_id": document_id,
                "limit": limit,
                "candidates": candidates,
            },
        )

    def answer_question(
        self,
        question: str,
        candidates: int = 10,
        limit: int = 5,
        minimum_rerank_score: float | None = None,
        document_id: str | None = None,
        conversation_history: list[
            dict[str, str]
        ] | None = None,
    ) -> dict[str, Any]:
        """Generate a document-scoped grounded answer."""

        return self._request(
            method="POST",
            endpoint="/api/v1/answers",
            payload={
                "question": question,
                "document_id": document_id,
                "conversation_history": (
                    conversation_history or []
                ),
                "candidates": candidates,
                "limit": limit,
                "minimum_rerank_score": (
                    minimum_rerank_score
                ),
            },
        )

    def process_document(
        self,
        path: str,
        chunk_size: int = 800,
        overlap: int = 120,
    ) -> dict[str, dict[str, Any]]:
        """Run ingestion, chunking, and indexing together."""

        ingestion = self.ingest_document(path)

        document_id = str(
            ingestion["document_id"]
        )

        chunking = self.chunk_document(
            document_id=document_id,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        indexing = self.index_document(
            document_id=document_id
        )

        return {
            "ingestion": ingestion,
            "chunking": chunking,
            "indexing": indexing,
        }