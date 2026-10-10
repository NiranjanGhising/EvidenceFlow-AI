"""Models returned by sparse and hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SparseSearchResult:
    """One result returned by BM25 retrieval."""

    chunk_id: str
    document_id: str
    source_name: str
    text: str
    score: float
    rank: int
    chunk_order: int
    heading_path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HybridSearchResult:
    """One result after sparse and dense ranking fusion."""

    chunk_id: str
    document_id: str
    source_name: str
    text: str
    fusion_score: float
    fusion_rank: int
    sparse_score: float | None
    sparse_rank: int | None
    dense_score: float | None
    dense_rank: int | None
    chunk_order: int
    heading_path: tuple[str, ...]