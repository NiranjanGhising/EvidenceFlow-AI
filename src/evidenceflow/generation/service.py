"""Evidence sufficiency and grounded answer generation."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from time import perf_counter

from evidenceflow.generation.citations import (
    build_citations,
    citation_anchor,
    validate_citations,
)
from evidenceflow.generation.models import GroundedAnswer
from evidenceflow.reranking.models import RerankedSearchResult
from evidenceflow.reranking.service import RerankedRetriever

logger = logging.getLogger(__name__)

REFUSAL_TEXT = (
    "I do not have enough evidence in the indexed documents "
    "to answer this question."
)


@dataclass(frozen=True, slots=True)
class EvidenceDecision:
    """Evidence-sufficiency decision."""

    sufficient: bool
    reason: str | None
    selected_evidence: tuple[RerankedSearchResult, ...]


class EvidenceSufficiencyChecker:
    """Decide whether retrieved evidence is sufficient."""

    def __init__(
        self,
        minimum_rerank_score: float = 0.0,
        maximum_evidence: int = 2,
    ) -> None:
        if maximum_evidence <= 0:
            raise ValueError(
                "maximum_evidence must be greater than zero"
            )

        self.minimum_rerank_score = minimum_rerank_score
        self.maximum_evidence = maximum_evidence

    def evaluate(
        self,
        results: list[RerankedSearchResult],
    ) -> EvidenceDecision:
        """Select evidence that meets the configured score threshold."""

        if not results:
            return EvidenceDecision(
                sufficient=False,
                reason="no_retrieval_results",
                selected_evidence=(),
            )

        selected = [
            result
            for result in results
            if result.rerank_score >= self.minimum_rerank_score
        ][: self.maximum_evidence]

        if not selected:
            return EvidenceDecision(
                sufficient=False,
                reason="rerank_score_below_threshold",
                selected_evidence=(),
            )

        return EvidenceDecision(
            sufficient=True,
            reason=None,
            selected_evidence=tuple(selected),
        )


def first_sentence(text: str) -> str:
    """Extract the first non-empty sentence from evidence text."""

    normalized = " ".join(text.split())

    if not normalized:
        return ""

    sentence_parts = re.split(
        r"(?<=[.!?])\s+",
        normalized,
        maxsplit=1,
    )

    return sentence_parts[0].strip()


class ExtractiveAnswerGenerator:
    """Generate a transparent answer directly from evidence."""

    def generate(
        self,
        evidence: list[RerankedSearchResult],
    ) -> str:
        """Create an extractive answer containing citation anchors."""

        answer_parts: list[str] = []

        for result in evidence:
            sentence = first_sentence(result.text)

            if not sentence:
                continue

            answer_parts.append(
                f"{sentence} {citation_anchor(result.chunk_id)}"
            )

        return " ".join(answer_parts).strip()


class GroundedAnswerService:
    """Retrieve, rerank, validate, and answer from evidence."""

    def __init__(
        self,
        retriever: RerankedRetriever,
        sufficiency_checker: EvidenceSufficiencyChecker,
        answer_generator: ExtractiveAnswerGenerator,
    ) -> None:
        self.retriever = retriever
        self.sufficiency_checker = sufficiency_checker
        self.answer_generator = answer_generator

    def answer(
        self,
        question: str,
        candidate_limit: int = 20,
        result_limit: int = 5,
    ) -> GroundedAnswer:
        """Generate a grounded answer or an explicit refusal."""

        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError("Question cannot be empty")

        total_start = perf_counter()

        retrieval_response = self.retriever.search(
            query=normalized_question,
            candidate_limit=candidate_limit,
            result_limit=result_limit,
        )

        retrieval_end = perf_counter()

        available_evidence = list(retrieval_response.results)

        decision = self.sufficiency_checker.evaluate(
            available_evidence
        )

        decision_end = perf_counter()

        if not decision.sufficient:
            citation_validation = validate_citations(
                answer=REFUSAL_TEXT,
                citations=[],
                available_evidence=available_evidence,
            )

            total_end = perf_counter()

            return GroundedAnswer(
                question=normalized_question,
                answer=REFUSAL_TEXT,
                supported=False,
                refusal_reason=decision.reason,
                citations=(),
                citation_validation=citation_validation,
                evidence=(),
                retrieval_mode="hybrid_rrf_cross_encoder",
                model_name=retrieval_response.model_name,
                used_reranker_fallback=(
                    retrieval_response.used_fallback
                ),
                latency_ms={
                    "retrieval": (
                        retrieval_end - total_start
                    )
                    * 1000,
                    "evidence_check": (
                        decision_end - retrieval_end
                    )
                    * 1000,
                    "generation": 0.0,
                    "citation_validation": 0.0,
                    "total": (
                        total_end - total_start
                    )
                    * 1000,
                },
                warnings=(),
            )

        selected_evidence = list(decision.selected_evidence)

        generation_start = perf_counter()

        generated_answer = self.answer_generator.generate(
            selected_evidence
        )

        generation_end = perf_counter()

        if generated_answer:
            supported = True
            refusal_reason = None
        else:
            generated_answer = REFUSAL_TEXT
            selected_evidence = []
            supported = False
            refusal_reason = "empty_generated_answer"

        citations = build_citations(selected_evidence)

        validation_start = perf_counter()

        citation_validation = validate_citations(
            answer=generated_answer,
            citations=citations,
            available_evidence=available_evidence,
        )

        validation_end = perf_counter()

        warnings: list[str] = []

        if supported and not citation_validation.valid:
            warnings.append("citation_validation_failed")

            supported = False
            refusal_reason = "invalid_citations"
            generated_answer = REFUSAL_TEXT
            selected_evidence = []
            citations = []

            citation_validation = validate_citations(
                answer=generated_answer,
                citations=[],
                available_evidence=available_evidence,
            )

        total_end = perf_counter()

        latency_ms = {
            "retrieval": (
                retrieval_end - total_start
            )
            * 1000,
            "evidence_check": (
                decision_end - retrieval_end
            )
            * 1000,
            "generation": (
                generation_end - generation_start
            )
            * 1000,
            "citation_validation": (
                validation_end - validation_start
            )
            * 1000,
            "total": (
                total_end - total_start
            )
            * 1000,
        }

        logger.info(
            "grounded_answer_completed",
            extra={
                "question": normalized_question,
                "supported": supported,
                "citation_count": len(citations),
                "evidence_count": len(selected_evidence),
                "refusal_reason": refusal_reason,
                "latency_ms": latency_ms["total"],
            },
        )

        return GroundedAnswer(
            question=normalized_question,
            answer=generated_answer,
            supported=supported,
            refusal_reason=refusal_reason,
            citations=tuple(citations),
            citation_validation=citation_validation,
            evidence=tuple(selected_evidence),
            retrieval_mode="hybrid_rrf_cross_encoder",
            model_name=retrieval_response.model_name,
            used_reranker_fallback=(
                retrieval_response.used_fallback
            ),
            latency_ms=latency_ms,
            warnings=tuple(warnings),
        )