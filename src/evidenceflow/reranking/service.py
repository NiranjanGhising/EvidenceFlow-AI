"""Cross-encoder reranking and retrieval orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter

from evidenceflow.reranking.encoder import PassageReranker
from evidenceflow.reranking.models import RerankedSearchResult
from evidenceflow.retrieval.models import HybridSearchResult
from evidenceflow.retrieval.service import (
    HybridRetriever,
    HybridSearchResponse,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RerankingResponse:
    """Reranked results with timing and fallback information."""

    query: str
    results: tuple[RerankedSearchResult, ...]
    hybrid_response: HybridSearchResponse
    model_name: str
    used_fallback: bool
    fallback_reason: str | None
    latency_ms: dict[str, float]


class RerankingService:
    """Rerank hybrid candidates using a passage-scoring model."""

    def __init__(
        self,
        reranker: PassageReranker,
        fallback_enabled: bool = True,
    ) -> None:
        self.reranker = reranker
        self.fallback_enabled = fallback_enabled

    def rerank(
        self,
        query: str,
        candidates: list[HybridSearchResult],
        top_k: int = 5,
    ) -> tuple[
        list[RerankedSearchResult],
        bool,
        str | None,
    ]:
        """Rerank candidates or preserve fusion order on failure."""

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("Reranking query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not candidates:
            return [], False, None

        try:
            scores = self.reranker.score(
                query=normalized_query,
                passages=[
                    candidate.text
                    for candidate in candidates
                ],
            )

            scored_candidates = sorted(
                zip(candidates, scores, strict=True),
                key=lambda item: (
                    -float(item[1]),
                    item[0].fusion_rank,
                    item[0].chunk_id,
                ),
            )

            results = [
                self._build_result(
                    candidate=candidate,
                    rerank_score=float(score),
                    rerank_rank=index,
                )
                for index, (candidate, score) in enumerate(
                    scored_candidates[:top_k],
                    start=1,
                )
            ]

            return results, False, None

        except Exception as exc:
            if not self.fallback_enabled:
                raise

            logger.exception(
                "reranker_failed_using_fusion_fallback",
                extra={
                    "model_name": self.reranker.model_name,
                    "candidate_count": len(candidates),
                },
            )

            fallback_results = [
                self._build_result(
                    candidate=candidate,
                    rerank_score=candidate.fusion_score,
                    rerank_rank=index,
                )
                for index, candidate in enumerate(
                    candidates[:top_k],
                    start=1,
                )
            ]

            return (
                fallback_results,
                True,
                f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _build_result(
        candidate: HybridSearchResult,
        rerank_score: float,
        rerank_rank: int,
    ) -> RerankedSearchResult:
        """Convert a hybrid candidate into a reranked result."""

        return RerankedSearchResult(
            chunk_id=candidate.chunk_id,
            document_id=candidate.document_id,
            source_name=candidate.source_name,
            text=candidate.text,
            rerank_score=rerank_score,
            rerank_rank=rerank_rank,
            fusion_score=candidate.fusion_score,
            fusion_rank=candidate.fusion_rank,
            sparse_score=candidate.sparse_score,
            sparse_rank=candidate.sparse_rank,
            dense_score=candidate.dense_score,
            dense_rank=candidate.dense_rank,
            chunk_order=candidate.chunk_order,
            heading_path=candidate.heading_path,
        )


class RerankedRetriever:
    """Run hybrid retrieval followed by cross-encoder reranking."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranking_service: RerankingService,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.reranking_service = reranking_service

    def search(
        self,
        query: str,
        candidate_limit: int = 20,
        result_limit: int = 5,
        dense_minimum_score: float | None = None,
        document_id: str | None = None,
    ) -> RerankingResponse:
        """Retrieve document-scoped candidates and rerank them."""

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "Search query cannot be empty"
            )

        if candidate_limit <= 0:
            raise ValueError(
                "candidate_limit must be greater than zero"
            )

        if result_limit <= 0:
            raise ValueError(
                "result_limit must be greater than zero"
            )

        if result_limit > candidate_limit:
            raise ValueError(
                "result_limit cannot exceed candidate_limit"
            )

        normalized_document_id = None

        if document_id is not None:
            normalized_document_id = document_id.strip()

            if not normalized_document_id:
                raise ValueError(
                    "document_id cannot be empty"
                )

        total_start = perf_counter()

        hybrid_response = self.hybrid_retriever.search(
            query=normalized_query,
            candidate_limit=candidate_limit,
            result_limit=candidate_limit,
            dense_minimum_score=dense_minimum_score,
            document_id=normalized_document_id,
        )

        hybrid_end = perf_counter()

        (
            reranked_results,
            used_fallback,
            fallback_reason,
        ) = self.reranking_service.rerank(
            query=normalized_query,
            candidates=list(
                hybrid_response.results
            ),
            top_k=result_limit,
        )

        rerank_end = perf_counter()

        latency_ms = {
            "hybrid": (
                hybrid_end - total_start
            )
            * 1000,
            "rerank": (
                rerank_end - hybrid_end
            )
            * 1000,
            "total": (
                rerank_end - total_start
            )
            * 1000,
        }

        logger.info(
            "reranked_search_completed",
            extra={
                "query": normalized_query,
                "document_id": normalized_document_id,
                "candidate_count": len(
                    hybrid_response.results
                ),
                "result_count": len(
                    reranked_results
                ),
                "model_name": (
                    self.reranking_service
                    .reranker
                    .model_name
                ),
                "used_fallback": used_fallback,
                "latency_ms": latency_ms["total"],
            },
        )

        return RerankingResponse(
            query=normalized_query,
            results=tuple(
                reranked_results
            ),
            hybrid_response=hybrid_response,
            model_name=(
                self.reranking_service
                .reranker
                .model_name
            ),
            used_fallback=used_fallback,
            fallback_reason=fallback_reason,
            latency_ms=latency_ms,
        )