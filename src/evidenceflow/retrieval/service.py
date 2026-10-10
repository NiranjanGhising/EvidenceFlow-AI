"""Hybrid sparse and dense retrieval orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from evidenceflow.embeddings.models import SemanticSearchResult
from evidenceflow.retrieval.fusion import weighted_rrf
from evidenceflow.retrieval.models import (
    HybridSearchResult,
    SparseSearchResult,
)
from evidenceflow.retrieval.sparse import BM25Retriever

logger = logging.getLogger(__name__)


class DenseSearchService(Protocol):
    """Interface required from the dense retrieval service."""

    def search(
        self,
        query: str,
        limit: int = 5,
        minimum_score: float | None = None,
    ) -> list[SemanticSearchResult]:
        """Return semantic search results."""


@dataclass(frozen=True, slots=True)
class HybridSearchResponse:
    """Hybrid results and retrieval-stage latency."""

    query: str
    results: tuple[HybridSearchResult, ...]
    sparse_results: tuple[SparseSearchResult, ...]
    dense_results: tuple[SemanticSearchResult, ...]
    latency_ms: dict[str, float]


class HybridRetriever:
    """Run BM25 and dense search before weighted RRF."""

    def __init__(
        self,
        sparse_retriever: BM25Retriever,
        dense_retriever: DenseSearchService,
        sparse_weight: float = 1.0,
        dense_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> None:
        if sparse_weight < 0 or dense_weight < 0:
            raise ValueError(
                "retrieval weights cannot be negative"
            )

        if sparse_weight == 0 and dense_weight == 0:
            raise ValueError(
                "at least one retrieval weight must be positive"
            )

        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")

        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        candidate_limit: int = 20,
        result_limit: int = 5,
        dense_minimum_score: float | None = None,
    ) -> HybridSearchResponse:
        """Run sparse, dense, and fusion stages."""

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("Search query cannot be empty")

        if candidate_limit <= 0:
            raise ValueError(
                "candidate_limit must be greater than zero"
            )

        if result_limit <= 0:
            raise ValueError(
                "result_limit must be greater than zero"
            )

        sparse_start = perf_counter()

        sparse_results = self.sparse_retriever.search(
            query=normalized_query,
            limit=candidate_limit,
        )

        sparse_end = perf_counter()

        dense_results = self.dense_retriever.search(
            query=normalized_query,
            limit=candidate_limit,
            minimum_score=dense_minimum_score,
        )

        dense_end = perf_counter()

        results = weighted_rrf(
            sparse_results=sparse_results,
            dense_results=dense_results,
            limit=result_limit,
            rrf_k=self.rrf_k,
            sparse_weight=self.sparse_weight,
            dense_weight=self.dense_weight,
        )

        fusion_end = perf_counter()

        latency_ms = {
            "sparse": (sparse_end - sparse_start) * 1000,
            "dense": (dense_end - sparse_end) * 1000,
            "fusion": (fusion_end - dense_end) * 1000,
            "total": (fusion_end - sparse_start) * 1000,
        }

        logger.info(
            "hybrid_search_completed",
            extra={
                "query": normalized_query,
                "sparse_count": len(sparse_results),
                "dense_count": len(dense_results),
                "result_count": len(results),
                "latency_ms": latency_ms["total"],
            },
        )

        return HybridSearchResponse(
            query=normalized_query,
            results=tuple(results),
            sparse_results=tuple(sparse_results),
            dense_results=tuple(dense_results),
            latency_ms=latency_ms,
        )