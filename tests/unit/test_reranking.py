import numpy as np

from evidenceflow.reranking.service import RerankingService
from evidenceflow.retrieval.models import HybridSearchResult


class FakeReranker:
    @property
    def model_name(self) -> str:
        return "fake-reranker"

    def score(
        self,
        query: str,
        passages: list[str],
    ) -> np.ndarray:
        del query

        return np.asarray(
            [
                10.0
                if "best evidence" in passage
                else 1.0
                for passage in passages
            ],
            dtype=np.float32,
        )


class FailingReranker:
    @property
    def model_name(self) -> str:
        return "failing-reranker"

    def score(
        self,
        query: str,
        passages: list[str],
    ) -> np.ndarray:
        del query
        del passages

        raise RuntimeError("model unavailable")


def candidate(
    chunk_id: str,
    text: str,
    fusion_rank: int,
) -> HybridSearchResult:
    return HybridSearchResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        source_name="guide.md",
        text=text,
        fusion_score=1.0 / fusion_rank,
        fusion_rank=fusion_rank,
        sparse_score=1.0,
        sparse_rank=fusion_rank,
        dense_score=0.5,
        dense_rank=fusion_rank,
        chunk_order=fusion_rank - 1,
        heading_path=("Guide",),
    )


def test_reranker_changes_candidate_order() -> None:
    service = RerankingService(
        reranker=FakeReranker()
    )

    candidates = [
        candidate(
            "chunk-a",
            "Ordinary supporting text.",
            1,
        ),
        candidate(
            "chunk-b",
            "This is the best evidence.",
            2,
        ),
    ]

    results, used_fallback, reason = service.rerank(
        query="Which passage is best?",
        candidates=candidates,
        top_k=2,
    )

    assert results[0].chunk_id == "chunk-b"
    assert results[0].fusion_rank == 2
    assert results[0].rerank_rank == 1
    assert used_fallback is False
    assert reason is None


def test_reranker_respects_top_k() -> None:
    service = RerankingService(
        reranker=FakeReranker()
    )

    results, _, _ = service.rerank(
        query="test query",
        candidates=[
            candidate("chunk-a", "first", 1),
            candidate("chunk-b", "second", 2),
        ],
        top_k=1,
    )

    assert len(results) == 1


def test_reranker_falls_back_to_fusion_order() -> None:
    service = RerankingService(
        reranker=FailingReranker(),
        fallback_enabled=True,
    )

    candidates = [
        candidate("chunk-a", "first", 1),
        candidate("chunk-b", "second", 2),
    ]

    results, used_fallback, reason = service.rerank(
        query="test query",
        candidates=candidates,
        top_k=2,
    )

    assert [
        result.chunk_id
        for result in results
    ] == [
        "chunk-a",
        "chunk-b",
    ]

    assert used_fallback is True
    assert reason is not None
    assert "model unavailable" in reason


def test_empty_candidates_return_empty_results() -> None:
    service = RerankingService(
        reranker=FakeReranker()
    )

    results, used_fallback, reason = service.rerank(
        query="test query",
        candidates=[],
        top_k=5,
    )

    assert results == []
    assert used_fallback is False
    assert reason is None