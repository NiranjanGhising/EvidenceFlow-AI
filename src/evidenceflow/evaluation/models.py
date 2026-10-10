"""Data models used by the evaluation framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationQuestion:
    """One manually reviewed evaluation question."""

    question_id: str
    question: str
    relevant_chunk_ids: frozenset[str]
    answerable: bool
    expected_citation_ids: frozenset[str] = frozenset()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievalEvaluation:
    """Retrieval metrics calculated for one question."""

    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    retrieved_chunk_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AnswerEvaluation:
    """Answer and citation metrics for one question."""

    supported: bool
    expected_answerable: bool
    refusal_correct: bool
    unsupported_answer: bool
    citation_valid: bool
    citation_coverage: float
    returned_citation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QuestionEvaluationResult:
    """Complete diagnostics for one evaluation question."""

    question_id: str
    question: str
    retrieval: RetrievalEvaluation
    answer: AnswerEvaluation
    latency_ms: dict[str, float]
    passed: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Aggregate metrics for one evaluation run."""

    question_count: int
    precision_at_k: float
    recall_at_k: float
    mean_reciprocal_rank: float
    mean_ndcg_at_k: float
    citation_validity_rate: float
    mean_citation_coverage: float
    refusal_accuracy: float
    unsupported_answer_rate: float
    mean_latency_ms: float
    passed_question_rate: float


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Complete evaluation output with provenance."""

    run_id: str
    created_at: str
    retrieval_limit: int
    embedding_model: str
    reranker_model: str
    summary: EvaluationSummary
    results: tuple[QuestionEvaluationResult, ...]