"""Evaluation runner for retrieval and grounded answers."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from evidenceflow.evaluation.metrics import (
    citation_coverage,
    is_unsupported_answer,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    refusal_is_correct,
    safe_mean,
)
from evidenceflow.evaluation.models import (
    AnswerEvaluation,
    EvaluationQuestion,
    EvaluationReport,
    EvaluationSummary,
    QuestionEvaluationResult,
    RetrievalEvaluation,
)
from evidenceflow.generation.service import GroundedAnswerService


class EvaluationRunner:
    """Evaluate retrieval, answers, citations, refusal, and latency."""

    def __init__(
        self,
        answer_service: GroundedAnswerService,
        embedding_model: str,
        reranker_model: str,
        retrieval_limit: int = 5,
        candidate_limit: int = 10,
    ) -> None:
        if retrieval_limit <= 0:
            raise ValueError(
                "retrieval_limit must be greater than zero"
            )

        if candidate_limit <= 0:
            raise ValueError(
                "candidate_limit must be greater than zero"
            )

        if retrieval_limit > candidate_limit:
            raise ValueError(
                "retrieval_limit cannot exceed candidate_limit"
            )

        self.answer_service = answer_service
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.retrieval_limit = retrieval_limit
        self.candidate_limit = candidate_limit

    def evaluate_question(
        self,
        question: EvaluationQuestion,
    ) -> QuestionEvaluationResult:
        """Evaluate one question and return detailed diagnostics."""

        answer_result = self.answer_service.answer(
            question=question.question,
            candidate_limit=self.candidate_limit,
            result_limit=self.retrieval_limit,
        )

        retrieved_chunk_ids = tuple(
            evidence.chunk_id
            for evidence in answer_result.evidence
        )

        retrieval_evaluation = RetrievalEvaluation(
            precision_at_k=precision_at_k(
                retrieved_ids=retrieved_chunk_ids,
                relevant_ids=question.relevant_chunk_ids,
                k=self.retrieval_limit,
            ),
            recall_at_k=recall_at_k(
                retrieved_ids=retrieved_chunk_ids,
                relevant_ids=question.relevant_chunk_ids,
                k=self.retrieval_limit,
            ),
            reciprocal_rank=reciprocal_rank(
                retrieved_ids=retrieved_chunk_ids,
                relevant_ids=question.relevant_chunk_ids,
            ),
            ndcg_at_k=ndcg_at_k(
                retrieved_ids=retrieved_chunk_ids,
                relevant_ids=question.relevant_chunk_ids,
                k=self.retrieval_limit,
            ),
            retrieved_chunk_ids=retrieved_chunk_ids,
        )

        returned_citation_ids = tuple(
            citation.citation_id
            for citation in answer_result.citations
        )

        expected_citation_ids = (
            question.expected_citation_ids
            if question.expected_citation_ids
            else question.relevant_chunk_ids
        )

        refusal_correct = refusal_is_correct(
            supported=answer_result.supported,
            expected_answerable=question.answerable,
        )

        unsupported_answer = is_unsupported_answer(
            supported=answer_result.supported,
            expected_answerable=question.answerable,
        )

        answer_evaluation = AnswerEvaluation(
            supported=answer_result.supported,
            expected_answerable=question.answerable,
            refusal_correct=refusal_correct,
            unsupported_answer=unsupported_answer,
            citation_valid=(
                answer_result.citation_validation.valid
            ),
            citation_coverage=citation_coverage(
                returned_citation_ids=returned_citation_ids,
                expected_citation_ids=expected_citation_ids,
            ),
            returned_citation_ids=returned_citation_ids,
        )

        failure_reasons: list[str] = []

        if question.answerable:
            if retrieval_evaluation.recall_at_k < 1.0:
                failure_reasons.append(
                    "relevant_chunks_not_fully_retrieved"
                )

            if not answer_result.supported:
                failure_reasons.append(
                    "answerable_question_was_refused"
                )

            if not answer_evaluation.citation_valid:
                failure_reasons.append(
                    "citation_validation_failed"
                )

            if answer_evaluation.citation_coverage < 1.0:
                failure_reasons.append(
                    "expected_citations_not_fully_covered"
                )
        else:
            if unsupported_answer:
                failure_reasons.append(
                    "unsupported_answer_returned"
                )

            if not refusal_correct:
                failure_reasons.append(
                    "unanswerable_question_not_refused"
                )

        return QuestionEvaluationResult(
            question_id=question.question_id,
            question=question.question,
            retrieval=retrieval_evaluation,
            answer=answer_evaluation,
            latency_ms=dict(answer_result.latency_ms),
            passed=not failure_reasons,
            failure_reasons=tuple(failure_reasons),
        )

    def run(
        self,
        questions: list[EvaluationQuestion],
    ) -> EvaluationReport:
        """Run evaluation over all questions."""

        if not questions:
            raise ValueError(
                "questions cannot be empty"
            )

        results = tuple(
            self.evaluate_question(question)
            for question in questions
        )

        summary = EvaluationSummary(
            question_count=len(results),
            precision_at_k=safe_mean(
                [
                    result.retrieval.precision_at_k
                    for result in results
                ]
            ),
            recall_at_k=safe_mean(
                [
                    result.retrieval.recall_at_k
                    for result in results
                ]
            ),
            mean_reciprocal_rank=safe_mean(
                [
                    result.retrieval.reciprocal_rank
                    for result in results
                ]
            ),
            mean_ndcg_at_k=safe_mean(
                [
                    result.retrieval.ndcg_at_k
                    for result in results
                ]
            ),
            citation_validity_rate=safe_mean(
                [
                    float(result.answer.citation_valid)
                    for result in results
                ]
            ),
            mean_citation_coverage=safe_mean(
                [
                    result.answer.citation_coverage
                    for result in results
                ]
            ),
            refusal_accuracy=safe_mean(
                [
                    float(result.answer.refusal_correct)
                    for result in results
                ]
            ),
            unsupported_answer_rate=safe_mean(
                [
                    float(
                        result.answer.unsupported_answer
                    )
                    for result in results
                ]
            ),
            mean_latency_ms=safe_mean(
                [
                    result.latency_ms.get(
                        "total",
                        0.0,
                    )
                    for result in results
                ]
            ),
            passed_question_rate=safe_mean(
                [
                    float(result.passed)
                    for result in results
                ]
            ),
        )

        return EvaluationReport(
            run_id=f"eval-{uuid4().hex[:12]}",
            created_at=datetime.now(UTC).isoformat(),
            retrieval_limit=self.retrieval_limit,
            embedding_model=self.embedding_model,
            reranker_model=self.reranker_model,
            summary=summary,
            results=results,
        )