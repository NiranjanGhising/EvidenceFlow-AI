"""Weighted Reciprocal Rank Fusion for sparse and dense rankings."""

from __future__ import annotations

from collections import defaultdict

from evidenceflow.embeddings.models import SemanticSearchResult
from evidenceflow.retrieval.models import (
    HybridSearchResult,
    SparseSearchResult,
)


def weighted_rrf(
    sparse_results: list[SparseSearchResult],
    dense_results: list[SemanticSearchResult],
    limit: int = 10,
    rrf_k: int = 60,
    sparse_weight: float = 1.0,
    dense_weight: float = 1.0,
) -> list[HybridSearchResult]:
    """Fuse sparse and dense rankings using weighted RRF."""

    if limit <= 0:
        raise ValueError("limit must be greater than zero")

    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero")

    if sparse_weight < 0 or dense_weight < 0:
        raise ValueError("fusion weights cannot be negative")

    if sparse_weight == 0 and dense_weight == 0:
        raise ValueError(
            "at least one fusion weight must be greater than zero"
        )

    fusion_scores: dict[str, float] = defaultdict(float)
    metadata: dict[str, dict[str, object]] = {}

    for rank, result in enumerate(
        sparse_results,
        start=1,
    ):
        fusion_scores[result.chunk_id] += (
            sparse_weight / (rrf_k + rank)
        )

        metadata[result.chunk_id] = {
            "chunk_id": result.chunk_id,
            "document_id": result.document_id,
            "source_name": result.source_name,
            "text": result.text,
            "chunk_order": result.chunk_order,
            "heading_path": result.heading_path,
            "sparse_score": result.score,
            "sparse_rank": rank,
            "dense_score": None,
            "dense_rank": None,
        }

    for rank, result in enumerate(
        dense_results,
        start=1,
    ):
        fusion_scores[result.chunk_id] += (
            dense_weight / (rrf_k + rank)
        )

        entry = metadata.setdefault(
            result.chunk_id,
            {
                "chunk_id": result.chunk_id,
                "document_id": result.document_id,
                "source_name": result.source_name,
                "text": result.text,
                "chunk_order": result.chunk_order,
                "heading_path": result.heading_path,
                "sparse_score": None,
                "sparse_rank": None,
                "dense_score": None,
                "dense_rank": None,
            },
        )

        entry["dense_score"] = result.score
        entry["dense_rank"] = rank

    ranked_chunk_ids = sorted(
        fusion_scores,
        key=lambda chunk_id: (
            -fusion_scores[chunk_id],
            chunk_id,
        ),
    )

    results: list[HybridSearchResult] = []

    for fusion_rank, chunk_id in enumerate(
        ranked_chunk_ids[:limit],
        start=1,
    ):
        entry = metadata[chunk_id]

        results.append(
            HybridSearchResult(
                chunk_id=str(entry["chunk_id"]),
                document_id=str(entry["document_id"]),
                source_name=str(entry["source_name"]),
                text=str(entry["text"]),
                fusion_score=fusion_scores[chunk_id],
                fusion_rank=fusion_rank,
                sparse_score=(
                    float(entry["sparse_score"])
                    if entry["sparse_score"] is not None
                    else None
                ),
                sparse_rank=(
                    int(entry["sparse_rank"])
                    if entry["sparse_rank"] is not None
                    else None
                ),
                dense_score=(
                    float(entry["dense_score"])
                    if entry["dense_score"] is not None
                    else None
                ),
                dense_rank=(
                    int(entry["dense_rank"])
                    if entry["dense_rank"] is not None
                    else None
                ),
                chunk_order=int(entry["chunk_order"]),
                heading_path=tuple(entry["heading_path"]),
            )
        )

    return results