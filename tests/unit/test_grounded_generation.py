from evidenceflow.generation.citations import (
    build_citations,
    citation_anchor,
    extract_citation_ids,
    validate_citations,
)
from evidenceflow.generation.service import (
    EvidenceSufficiencyChecker,
    ExtractiveAnswerGenerator,
    first_sentence,
)
from evidenceflow.reranking.models import RerankedSearchResult


def make_evidence(
    chunk_id: str = "chunk-abc123",
    score: float = 5.0,
    text: str = "FastAPI validates requests.",
) -> RerankedSearchResult:
    return RerankedSearchResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        source_name="guide.md",
        text=text,
        rerank_score=score,
        rerank_rank=1,
        fusion_score=0.03,
        fusion_rank=1,
        sparse_score=1.0,
        sparse_rank=1,
        dense_score=0.8,
        dense_rank=1,
        chunk_order=0,
        heading_path=("Guide",),
    )


def test_citation_anchor() -> None:
    assert citation_anchor(
        "chunk-abc123"
    ) == "[chunk-abc123]"


def test_extract_citation_ids_removes_duplicates() -> None:
    answer = (
        "First [chunk-a]. "
        "Second [chunk-a]. "
        "Third [chunk-b]."
    )

    assert extract_citation_ids(answer) == [
        "chunk-a",
        "chunk-b",
    ]


def test_valid_citation_passes_validation() -> None:
    selected = [make_evidence()]
    citations = build_citations(selected)

    result = validate_citations(
        answer=(
            "FastAPI validates requests. "
            "[chunk-abc123]"
        ),
        citations=citations,
        available_evidence=selected,
    )

    assert result.valid is True
    assert result.coverage == 1.0
    assert result.invalid_citation_ids == ()


def test_fabricated_citation_is_rejected() -> None:
    selected = [make_evidence()]
    citations = build_citations(selected)

    result = validate_citations(
        answer="Unsupported claim [chunk-fake].",
        citations=citations,
        available_evidence=selected,
    )

    assert result.valid is False
    assert result.invalid_citation_ids == (
        "chunk-fake",
    )


def test_missing_citation_is_rejected() -> None:
    selected = [make_evidence()]
    citations = build_citations(selected)

    result = validate_citations(
        answer="FastAPI validates requests.",
        citations=citations,
        available_evidence=selected,
    )

    assert result.valid is False
    assert result.missing_citation_ids == (
        "chunk-abc123",
    )
    assert result.coverage == 0.0


def test_sufficiency_rejects_empty_results() -> None:
    checker = EvidenceSufficiencyChecker(
        minimum_rerank_score=0.0
    )

    decision = checker.evaluate([])

    assert decision.sufficient is False
    assert decision.reason == "no_retrieval_results"


def test_sufficiency_rejects_low_score() -> None:
    checker = EvidenceSufficiencyChecker(
        minimum_rerank_score=2.0
    )

    decision = checker.evaluate(
        [make_evidence(score=1.0)]
    )

    assert decision.sufficient is False
    assert decision.reason == (
        "rerank_score_below_threshold"
    )


def test_sufficiency_accepts_high_score() -> None:
    checker = EvidenceSufficiencyChecker(
        minimum_rerank_score=2.0
    )

    decision = checker.evaluate(
        [make_evidence(score=3.0)]
    )

    assert decision.sufficient is True
    assert len(decision.selected_evidence) == 1


def test_first_sentence() -> None:
    assert first_sentence(
        "First sentence. Second sentence."
    ) == "First sentence."


def test_extractive_answer_contains_citation() -> None:
    generator = ExtractiveAnswerGenerator()

    result = generator.generate(
        question="How does FastAPI validate requests?",
        evidence=[make_evidence()],
    )

    answer = result.text
    assert "FastAPI validates requests." in answer
    assert "[chunk-abc123]" in answer