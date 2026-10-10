"""Embedding and semantic-search result models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SemanticSearchResult:
    """One semantic-search result from the persistent vector index."""

    chunk_id: str
    document_id: str
    source_name: str
    text: str
    score: float
    chunk_order: int
    heading_path: tuple[str, ...]