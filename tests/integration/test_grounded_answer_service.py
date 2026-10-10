from dataclasses import dataclass

from evidenceflow.generation.service import (
    REFUSAL_TEXT,
    EvidenceSufficiencyChecker,
    ExtractiveAnswerGenerator,
    GroundedAnswerService,
)
from evidenceflow.reranking.models import RerankedSearchResult
from evidenceflow.reranking.service import RerankingResponse
from evidenceflow.retrieval.service import HybridSearchResponse


@dataclass
class FakeRetriever:
    results: tuple[RerankedSearchResult, ...]

    def search(
        self,
        query: str,
        candidate_limit: int = 20,
        result_limit: int = 5,
        dense_minimum_score: float | None = None,
        document_id: str | None = None,
    ) -> RerankingResponse:
        del (
            candidate_limit,
            result_limit,
            dense_minimum_score,
            document_id,
        )

        hybrid_response = HybridSearchResponse(
            query=query,
            results=(),
            sparse_results=(),
            dense_results=(),
            latency_ms={
                "sparse": 0.0,
                "dense": 0.0,
                "fusion": 0.0,
                "total": 0.0,
            },
        )

        return RerankingResponse(
            query=query,
            results=self.results,
            hybrid_response=hybrid_response,
            model_name="fake-reranker",
            used_fallback=False,
            fallback_reason=None,
            latency_ms={
                "hybrid": 0.0,
                "rerank": 0.0,
                "total": 0.0,
            },
        )


def make_evidence(
    score: float = 5.0,
) -> RerankedSearchResult:
    return RerankedSearchResult(
        chunk_id="chunk-abc123",
        document_id="doc-1",
        source_name="guide.md",
        text=(
            "FastAPI validates requests using "
            "Pydantic models."
        ),
        rerank_score=score,
        rerank_rank=1,
        fusion_score=0.03,
        fusion_rank=1,
        sparse_score=1.0,
        sparse_rank=1,
        dense_score=0.8,
        dense_rank=1,
        chunk_order=0,
        heading_path=("FastAPI Guide",),
    )


def build_grounded_service(
    results: tuple[RerankedSearchResult, ...],
    threshold: float = 0.0,
) -> GroundedAnswerService:
    return GroundedAnswerService(
        retriever=FakeRetriever(results),
        sufficiency_checker=(
            EvidenceSufficiencyChecker(
                minimum_rerank_score=threshold,
                maximum_evidence=2,
            )
        ),
        answer_generator=ExtractiveAnswerGenerator(),
    )


def test_supported_answer_contains_valid_citation() -> None:
    service = build_grounded_service(
        results=(make_evidence(score=5.0),),
        threshold=1.0,
    )

    result = service.answer(
        "How does FastAPI validate requests?"
    )

    assert result.supported is True
    assert result.refusal_reason is None
    assert "[chunk-abc123]" in result.answer
    assert len(result.citations) == 1
    assert result.citation_validation.valid is True
    assert result.citation_validation.coverage == 1.0


def test_missing_evidence_returns_refusal() -> None:
    service = build_grounded_service(
        results=(),
        threshold=1.0,
    )

    result = service.answer(
        "What does the document say?"
    )

    assert result.supported is False
    assert result.answer == REFUSAL_TEXT
    assert result.refusal_reason == "no_retrieval_results"
    assert result.citations == ()


def test_low_score_returns_refusal() -> None:
    service = build_grounded_service(
        results=(make_evidence(score=0.5),),
        threshold=1.0,
    )

    result = service.answer(
        "How does FastAPI validate requests?"
    )

    assert result.supported is False
    assert result.answer == REFUSAL_TEXT
    assert result.refusal_reason == (
        "rerank_score_below_threshold"
    )


def test_answer_reports_all_latency_stages() -> None:
    service = build_grounded_service(
        results=(make_evidence(score=5.0),),
        threshold=1.0,
    )

    result = service.answer(
        "How does FastAPI validate requests?"
    )

    assert set(result.latency_ms) == {
        "retrieval",
        "evidence_check",
        "generation",
        "citation_validation",
        "total",
    }

    assert result.latency_ms["total"] >= 0


def test_empty_question_is_rejected() -> None:
    service = build_grounded_service(
        results=(),
    )

    try:
        service.answer("   ")
    except ValueError as exc:
        assert "cannot be empty" in str(exc)
    else:
        raise AssertionError(
            "Empty question was not rejected"
        )