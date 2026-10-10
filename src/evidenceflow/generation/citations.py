"""Citation anchor creation, parsing, and validation."""

from __future__ import annotations

import re

from evidenceflow.generation.models import (
    Citation,
    CitationValidationResult,
)
from evidenceflow.reranking.models import RerankedSearchResult

CITATION_PATTERN = re.compile(
    r"\[(?P<citation_id>chunk-[A-Za-z0-9_-]+)\]"
)


def citation_anchor(chunk_id: str) -> str:
    """Return the visible citation anchor for a chunk."""

    if not chunk_id.strip():
        raise ValueError("chunk_id cannot be empty")

    return f"[{chunk_id}]"


def extract_citation_ids(answer: str) -> list[str]:
    """Extract citation IDs without returning duplicates."""


    citation_ids: list[str] = []
    seen: set[str] = set()

    for match in CITATION_PATTERN.finditer(answer):
        citation_id = match.group("citation_id")

        if citation_id in seen:
            continue

        seen.add(citation_id)
        citation_ids.append(citation_id)

    return citation_ids


def build_citations(
    evidence: list[RerankedSearchResult],
) -> list[Citation]:
    """Build structured citations from selected evidence."""

    return [
        Citation(
            citation_id=result.chunk_id,
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            source_name=result.source_name,
            heading_path=result.heading_path,
        )
        for result in evidence
    ]


def validate_citations(
    answer: str,
    citations: list[Citation],
    available_evidence: list[RerankedSearchResult],
) -> CitationValidationResult:
    """Validate citations against declared and available evidence."""

    answer_citation_ids = extract_citation_ids(answer)

    declared_ids = {
        citation.citation_id
        for citation in citations
    }

    available_ids = {
        result.chunk_id
        for result in available_evidence
    }

    valid_ids = [
        citation_id
        for citation_id in answer_citation_ids
        if citation_id in declared_ids
        and citation_id in available_ids
    ]

    invalid_ids = [
        citation_id
        for citation_id in answer_citation_ids
        if citation_id not in declared_ids
        or citation_id not in available_ids
    ]

    missing_ids = [
        citation_id
        for citation_id in declared_ids
        if citation_id not in answer_citation_ids
    ]

    if declared_ids:
        coverage = len(set(valid_ids)) / len(declared_ids)
    else:
        coverage = 1.0 if not answer_citation_ids else 0.0

    valid = (
        not invalid_ids
        and not missing_ids
        and coverage == 1.0
    )

    return CitationValidationResult(
        valid=valid,
        valid_citation_ids=tuple(valid_ids),
        invalid_citation_ids=tuple(invalid_ids),
        missing_citation_ids=tuple(sorted(missing_ids)),
        coverage=coverage,
    )