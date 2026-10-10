"""Models used by grounded answer generation."""

from __future__ import annotations

from dataclasses import dataclass

from evidenceflow.reranking.models import RerankedSearchResult


@dataclass(frozen=True, slots=True)
class Citation:
    """Citation attached to a grounded answer."""

    citation_id: str
    chunk_id: str
    document_id: str
    source_name: str
    heading_path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CitationValidationResult:
    """Result of independent citation validation."""

    valid: bool
    valid_citation_ids: tuple[str, ...]
    invalid_citation_ids: tuple[str, ...]
    missing_citation_ids: tuple[str, ...]
    coverage: float


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    """Grounded answer with support and citation information."""

    question: str
    answer: str
    supported: bool
    refusal_reason: str | None
    citations: tuple[Citation, ...]
    citation_validation: CitationValidationResult
    evidence: tuple[RerankedSearchResult, ...]
    retrieval_mode: str
    model_name: str
    used_reranker_fallback: bool
    latency_ms: dict[str, float]
    warnings: tuple[str, ...]
    generation_provider: str = "extractive"
    generation_model: str = "extractive"
    used_llm_fallback: bool = False
    llm_fallback_reason: str | None = None