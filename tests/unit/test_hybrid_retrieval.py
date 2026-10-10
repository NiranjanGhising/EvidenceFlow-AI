from evidenceflow.embeddings.models import (
    SemanticSearchResult,
)
from evidenceflow.retrieval.fusion import weighted_rrf
from evidenceflow.retrieval.models import SparseSearchResult
from evidenceflow.retrieval.tokenizer import tokenize_for_search


def sparse_result(
    chunk_id: str,
    score: float,
    rank: int,
) -> SparseSearchResult:
    return SparseSearchResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        source_name="guide.md",
        text=f"Text for {chunk_id}",
        score=score,
        rank=rank,
        chunk_order=rank - 1,
        heading_path=("Guide",),
    )


def dense_result(
    chunk_id: str,
    score: float,
) -> SemanticSearchResult:
    return SemanticSearchResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        source_name="guide.md",
        text=f"Text for {chunk_id}",
        score=score,
        chunk_order=0,
        heading_path=("Guide",),
    )


def test_tokenizer_preserves_technical_identifiers() -> None:
    tokens = tokenize_for_search(
        "FastAPI uses check_permission and api/v1/users."
    )

    assert "fastapi" in tokens
    assert "check_permission" in tokens
    assert "api/v1/users." in tokens


def test_rrf_promotes_result_present_in_both_lists() -> None:
    sparse = [
        sparse_result("chunk-a", 5.0, 1),
        sparse_result("chunk-b", 4.0, 2),
    ]

    dense = [
        dense_result("chunk-b", 0.9),
        dense_result("chunk-c", 0.8),
    ]

    results = weighted_rrf(
        sparse_results=sparse,
        dense_results=dense,
        limit=3,
    )

    assert results[0].chunk_id == "chunk-b"
    assert results[0].sparse_rank == 2
    assert results[0].dense_rank == 1


def test_rrf_does_not_duplicate_chunks() -> None:
    results = weighted_rrf(
        sparse_results=[
            sparse_result("chunk-a", 2.0, 1)
        ],
        dense_results=[
            dense_result("chunk-a", 0.8)
        ],
        limit=5,
    )

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-a"


def test_rrf_supports_sparse_only_result() -> None:
    results = weighted_rrf(
        sparse_results=[
            sparse_result("chunk-a", 2.0, 1)
        ],
        dense_results=[],
        limit=5,
    )

    assert len(results) == 1
    assert results[0].sparse_rank == 1
    assert results[0].dense_rank is None


def test_rrf_tie_break_is_deterministic() -> None:
    results = weighted_rrf(
        sparse_results=[
            sparse_result("chunk-b", 2.0, 1)
        ],
        dense_results=[
            dense_result("chunk-a", 0.8)
        ],
        limit=5,
    )

    assert [
        result.chunk_id for result in results
    ] == [
        "chunk-a",
        "chunk-b",
    ]