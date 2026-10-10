"""Evidence sufficiency and grounded answer generation."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from evidenceflow.generation.citations import (
    build_citations,
    citation_anchor,
    extract_citation_ids,
    validate_citations,
)
from evidenceflow.generation.models import GroundedAnswer
from evidenceflow.llm.models import LLMGenerationResult
from evidenceflow.reranking.models import RerankedSearchResult
from evidenceflow.reranking.service import RerankedRetriever

logger = logging.getLogger(__name__)

REFUSAL_TEXT = (
    "I do not have enough evidence in the indexed documents "
    "to answer this question."
)


class AnswerGenerator(Protocol):
    """Interface implemented by grounded answer generators."""

    @property
    def provider_name(self) -> str:
        """Return the generation provider name."""

    @property
    def model_name(self) -> str:
        """Return the generation model name."""

    def generate(
        self,
        question: str,
        evidence: list[RerankedSearchResult],
        conversation_history: list[
            dict[str, str]
        ] | None = None,
    ) -> LLMGenerationResult:
        """Generate an answer from retrieved evidence."""


@dataclass(frozen=True, slots=True)
class EvidenceDecision:
    """Evidence-sufficiency decision."""

    sufficient: bool
    reason: str | None
    selected_evidence: tuple[RerankedSearchResult, ...]

class EvidenceSufficiencyChecker:
    """Select relevant evidence using retrieval signals."""

    def __init__(
        self,
        minimum_rerank_score: float | None = None,
        minimum_dense_score: float = 0.20,
        maximum_evidence: int = 3,
    ) -> None:
        if maximum_evidence <= 0:
            raise ValueError(
                "maximum_evidence must be greater than zero"
            )

        if not 0.0 <= minimum_dense_score <= 1.0:
            raise ValueError(
                "minimum_dense_score must be between 0 and 1"
            )

        self.minimum_rerank_score = (
            minimum_rerank_score
        )
        self.minimum_dense_score = (
            minimum_dense_score
        )
        self.maximum_evidence = maximum_evidence

    def _has_retrieval_support(
        self,
        result: RerankedSearchResult,
    ) -> bool:
        """Return whether retrieval signals support the result."""

        has_sparse_support = (
            result.sparse_rank is not None
        )

        has_dense_support = (
            result.dense_score is not None
            and result.dense_score
            >= self.minimum_dense_score
        )

        return (
            has_sparse_support
            or has_dense_support
        )

    def evaluate(
        self,
        results: list[RerankedSearchResult],
    ) -> EvidenceDecision:
        """Select evidence using retrieval and reranking signals."""

        if not results:
            return EvidenceDecision(
                sufficient=False,
                reason="no_retrieval_results",
                selected_evidence=(),
            )

        supported_results = [
            result
            for result in results
            if self._has_retrieval_support(result)
        ]

        if not supported_results:
            return EvidenceDecision(
                sufficient=False,
                reason="no_supported_retrieval_results",
                selected_evidence=(),
            )

        if self.minimum_rerank_score is not None:
            supported_results = [
                result
                for result in supported_results
                if result.rerank_score
                >= self.minimum_rerank_score
            ]

            if not supported_results:
                return EvidenceDecision(
                    sufficient=False,
                    reason="rerank_score_below_threshold",
                    selected_evidence=(),
                )

        selected_evidence = tuple(
            supported_results[
                : self.maximum_evidence
            ]
        )

        return EvidenceDecision(
            sufficient=True,
            reason=None,
            selected_evidence=selected_evidence,
        )
def first_sentence(
    text: str,
) -> str:
    """Extract the first non-empty sentence from evidence text."""

    normalized = " ".join(
        text.split()
    )

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

    @property
    def provider_name(self) -> str:
        """Return the generator provider name."""

        return "evidenceflow"

    @property
    def model_name(self) -> str:
        """Return the generator model name."""

        return "extractive"

    def generate(
        self,
        question: str,
        evidence: list[RerankedSearchResult],
        conversation_history: list[
            dict[str, str]
        ] | None = None,
    ) -> LLMGenerationResult:
        """Create an extractive answer containing citations."""

        del question
        del conversation_history

        answer_parts: list[str] = []

        for result in evidence:
            sentence = first_sentence(
                result.text
            )

            if not sentence:
                continue

            answer_parts.append(
                f"{sentence} "
                f"{citation_anchor(result.chunk_id)}"
            )

        generated_text = " ".join(
            answer_parts
        ).strip()

        if not generated_text:
            generated_text = REFUSAL_TEXT

        return LLMGenerationResult(
            text=generated_text,
            provider=self.provider_name,
            model_name=self.model_name,
        )

class GroundedAnswerService:
    """Retrieve, rerank, generate, and validate an answer."""

    def __init__(
        self,
        retriever: RerankedRetriever,
        sufficiency_checker: EvidenceSufficiencyChecker,
        answer_generator: AnswerGenerator,
        fallback_generator: AnswerGenerator | None = None,
    ) -> None:
        self.retriever = retriever
        self.sufficiency_checker = sufficiency_checker
        self.answer_generator = answer_generator
        self.fallback_generator = fallback_generator

    def _generate_answer(
        self,
        question: str,
        evidence: list[RerankedSearchResult],
        conversation_history: list[
            dict[str, str]
        ] | None = None,
    ) -> tuple[LLMGenerationResult, list[str]]:
        
        """Generate an answer with optional extractive fallback."""

        warnings: list[str] = []

        try:
            generation_result = (
                self.answer_generator.generate(
                    question=question,
                    evidence=evidence,
                    conversation_history=(
                        conversation_history
                    ),
                )
            )

            return generation_result, warnings

        except Exception as exc:
            if self.fallback_generator is None:
                raise

            logger.exception(
                "llm_generation_failed_using_extractive_fallback",
                extra={
                    "provider": (
                        self.answer_generator.provider_name
                    ),
                    "model_name": (
                        self.answer_generator.model_name
                    ),
                },
            )

            warnings.append(
                "llm_generation_failed_using_extractive_fallback"
            )

            fallback_result = (
                self.fallback_generator.generate(
                    question=question,
                    evidence=evidence,
                    conversation_history=(
                        conversation_history
                    ),
                )
            )

            return (
                LLMGenerationResult(
                    text=fallback_result.text,
                    provider=fallback_result.provider,
                    model_name=(
                        fallback_result.model_name
                    ),
                    used_fallback=True,
                    fallback_reason=(
                        f"{type(exc).__name__}: {exc}"
                    ),
                ),
                warnings,
            )


    @staticmethod
    def _select_cited_evidence(
        answer: str,
        available_evidence: list[RerankedSearchResult],
    ) -> list[RerankedSearchResult]:
        """Select retrieved evidence referenced by the answer."""

        cited_ids = set(
            extract_citation_ids(answer)
        )

        return [
            result
            for result in available_evidence
            if result.chunk_id in cited_ids
        ]

    def answer(
        self,
        question: str,
        candidate_limit: int = 20,
        result_limit: int = 5,
        document_id: str | None = None,
        conversation_history: list[
            dict[str, str]
        ] | None = None,
    ) -> GroundedAnswer:
        """Generate a grounded answer or explicit refusal."""

        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError("Question cannot be empty")

        total_start = perf_counter()

        normalized_document_id = None

        if document_id is not None:
            normalized_document_id = document_id.strip()

            if not normalized_document_id:
                raise ValueError(
                    "document_id cannot be empty"
                )

        retrieval_response = self.retriever.search(
            query=normalized_question,
            candidate_limit=candidate_limit,
            result_limit=result_limit,
            document_id=normalized_document_id,
        )

        retrieval_end = perf_counter()

        available_evidence = list(
            retrieval_response.results
        )

        decision = self.sufficiency_checker.evaluate(
            available_evidence
        )

        decision_end = perf_counter()

        if not decision.sufficient:
            validation = validate_citations(
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
                citation_validation=validation,
                evidence=(),
                retrieval_mode=(
                    "hybrid_rrf_cross_encoder"
                ),
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
                generation_provider="evidenceflow",
                generation_model="retrieval-refusal",
            )

        selected_evidence = list(
            decision.selected_evidence
        )

        generation_start = perf_counter()

        generation_result, warnings = self._generate_answer(
            question=normalized_question,
            evidence=selected_evidence,
            conversation_history=conversation_history,
        )

        generation_end = perf_counter()

        generated_answer = generation_result.text.strip()

        if generated_answer == REFUSAL_TEXT:
            supported = False
            refusal_reason = "llm_evidence_refusal"
            cited_evidence: list[RerankedSearchResult] = []
        elif not generated_answer:
            supported = False
            refusal_reason = "empty_generated_answer"
            generated_answer = REFUSAL_TEXT
            cited_evidence = []
        else:
            supported = True
            refusal_reason = None
            cited_evidence = self._select_cited_evidence(
                answer=generated_answer,
                available_evidence=selected_evidence,
            )

        citations = build_citations(
            cited_evidence
        )

        validation_start = perf_counter()

        validation = validate_citations(
            answer=generated_answer,
            citations=citations,
            available_evidence=selected_evidence,
        )

        validation_end = perf_counter()

        if supported and not citations:
            warnings.append(
                "generated_answer_contains_no_citations"
            )

            supported = False
            refusal_reason = "missing_citations"

        if supported and not validation.valid:
            warnings.append(
                "citation_validation_failed"
            )

            supported = False
            refusal_reason = "invalid_citations"

        if not supported:
            generated_answer = REFUSAL_TEXT
            cited_evidence = []
            citations = []

            validation = validate_citations(
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
                "document_id": normalized_document_id,
                "supported": supported,
                "citation_count": len(citations),
                "evidence_count": len(cited_evidence),
                "generation_provider": (
                    generation_result.provider
                ),
                "generation_model": (
                    generation_result.model_name
                ),
                "used_llm_fallback": (
                    generation_result.used_fallback
                ),
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
            citation_validation=validation,
            evidence=tuple(cited_evidence),
            retrieval_mode="hybrid_rrf_cross_encoder",
            model_name=retrieval_response.model_name,
            used_reranker_fallback=(
                retrieval_response.used_fallback
            ),
            latency_ms=latency_ms,
            warnings=tuple(warnings),
            generation_provider=(
                generation_result.provider
            ),
            generation_model=(
                generation_result.model_name
            ),
            used_llm_fallback=(
                generation_result.used_fallback
            ),
            llm_fallback_reason=(
                generation_result.fallback_reason
            ),
        )