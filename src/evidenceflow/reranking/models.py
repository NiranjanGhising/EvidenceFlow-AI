"""Models returned by cross-encoder reranking."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RerankedSearchResult:
    """One result after cross-encoder reranking."""

    chunk_id: str
    document_id: str
    source_name: str
    text: str
    rerank_score: float
    rerank_rank: int
    fusion_score: float
    fusion_rank: int
    sparse_score: float | None
    sparse_rank: int | None
    dense_score: float | None
    dense_rank: int | None
    chunk_order: int
    heading_path: tuple[str, ...]