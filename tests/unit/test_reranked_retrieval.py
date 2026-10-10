from pathlib import Path

import numpy as np

from evidenceflow.chunking.service import ChunkingService
from evidenceflow.core.config import Settings
from evidenceflow.indexing.service import EmbeddingIndexService
from evidenceflow.ingestion.service import IngestionService
from evidenceflow.reranking.service import (
    RerankedRetriever,
    RerankingService,
)
from evidenceflow.retrieval.service import HybridRetriever
from evidenceflow.retrieval.sparse import BM25Retriever
from evidenceflow.storage.chunks import ChunkRepository
from evidenceflow.storage.documents import DocumentRepository
from evidenceflow.storage.vectors import VectorRepository


class FakeEncoder:
    @property
    def model_name(self) -> str:
        return "reranking-test-encoder"

    @property
    def dimension(self) -> int:
        return 4

    def encode(
        self,
        texts: list[str],
    ) -> np.ndarray:
        vectors: list[list[float]] = []

        for text in texts:
            lowered = text.lower()

            vectors.append(
                [
                    float(
                        "fastapi" in lowered
                        or "request" in lowered
                    ),
                    float(
                        "database" in lowered
                        or "sql" in lowered
                    ),
                    float(
                        "authorization" in lowered
                    ),
                    0.1,
                ]
            )

        array = np.asarray(
            vectors,
            dtype=np.float32,
        )

        norms = np.linalg.norm(
            array,
            axis=1,
            keepdims=True,
        )

        norms[norms == 0] = 1.0

        return array / norms


class FakePassageReranker:
    @property
    def model_name(self) -> str:
        return "reranking-test-model"

    def score(
        self,
        query: str,
        passages: list[str],
    ) -> np.ndarray:
        query_tokens = set(query.lower().split())
        scores: list[float] = []

        for passage in passages:
            passage_tokens = set(
                passage.lower().split()
            )

            scores.append(
                float(
                    len(
                        query_tokens.intersection(
                            passage_tokens
                        )
                    )
                )
            )

        return np.asarray(
            scores,
            dtype=np.float32,
        )


def build_retriever(
    tmp_path: Path,
) -> tuple[
    IngestionService,
    ChunkingService,
    EmbeddingIndexService,
    RerankedRetriever,
]:
    settings = Settings.from_env(tmp_path)

    document_repository = DocumentRepository(
        settings.database_path
    )

    ingestion_service = IngestionService(
        settings=settings,
        repository=document_repository,
    )

    chunk_repository = ChunkRepository(
        settings.database_path
    )

    chunking_service = ChunkingService(
        document_repository=document_repository,
        chunk_repository=chunk_repository,
        chunk_size=150,
        overlap=20,
    )

    vector_repository = VectorRepository(
        settings.database_path
    )

    indexing_service = EmbeddingIndexService(
        document_repository=document_repository,
        chunk_repository=chunk_repository,
        vector_repository=vector_repository,
        encoder=FakeEncoder(),
    )

    sparse_retriever = BM25Retriever(
        chunk_repository=chunk_repository,
    )

    hybrid_retriever = HybridRetriever(
        sparse_retriever=sparse_retriever,
        dense_retriever=indexing_service,
    )

    reranking_service = RerankingService(
        reranker=FakePassageReranker(),
    )

    reranked_retriever = RerankedRetriever(
        hybrid_retriever=hybrid_retriever,
        reranking_service=reranking_service,
    )

    return (
        ingestion_service,
        chunking_service,
        indexing_service,
        reranked_retriever,
    )


def add_document(
    ingestion_service: IngestionService,
    chunking_service: ChunkingService,
    indexing_service: EmbeddingIndexService,
    path: Path,
    text: str,
) -> None:
    path.write_text(text, encoding="utf-8")

    ingestion_result = ingestion_service.ingest(path)

    chunking_service.chunk_document(
        ingestion_result.document.document_id
    )

    indexing_service.index_document(
        ingestion_result.document.document_id
    )


def test_reranked_search_returns_relevant_result(
    tmp_path: Path,
) -> None:
    (
        ingestion_service,
        chunking_service,
        indexing_service,
        reranked_retriever,
    ) = build_retriever(tmp_path)

    add_document(
        ingestion_service,
        chunking_service,
        indexing_service,
        tmp_path / "api.md",
        (
            "# FastAPI Validation\n"
            "FastAPI validates request bodies using "
            "Pydantic models."
        ),
    )

    add_document(
        ingestion_service,
        chunking_service,
        indexing_service,
        tmp_path / "database.md",
        (
            "# Database\n"
            "PostgreSQL stores application records."
        ),
    )

    response = reranked_retriever.search(
        query="FastAPI validates request bodies",
        candidate_limit=10,
        result_limit=2,
    )

    assert response.results
    assert response.results[0].source_name == "api.md"
    assert response.results[0].rerank_rank == 1
    assert response.used_fallback is False


def test_reranked_result_preserves_rank_provenance(
    tmp_path: Path,
) -> None:
    (
        ingestion_service,
        chunking_service,
        indexing_service,
        reranked_retriever,
    ) = build_retriever(tmp_path)

    add_document(
        ingestion_service,
        chunking_service,
        indexing_service,
        tmp_path / "security.md",
        (
            "# Authorization\n"
            "Authorization protects restricted resources."
        ),
    )

    response = reranked_retriever.search(
        query="authorization restricted resources",
        candidate_limit=5,
        result_limit=1,
    )

    result = response.results[0]

    assert result.rerank_rank == 1
    assert result.fusion_rank >= 1
    assert result.sparse_rank is not None
    assert result.dense_rank is not None


def test_reranked_search_reports_latency(
    tmp_path: Path,
) -> None:
    (
        ingestion_service,
        chunking_service,
        indexing_service,
        reranked_retriever,
    ) = build_retriever(tmp_path)

    add_document(
        ingestion_service,
        chunking_service,
        indexing_service,
        tmp_path / "guide.txt",
        "FastAPI validates incoming request data.",
    )

    response = reranked_retriever.search(
        query="FastAPI request validation",
        candidate_limit=5,
        result_limit=1,
    )

    assert set(response.latency_ms) == {
        "hybrid",
        "rerank",
        "total",
    }

    assert response.latency_ms["total"] >= 0


def test_result_limit_cannot_exceed_candidates(
    tmp_path: Path,
) -> None:
    (
        _,
        _,
        _,
        reranked_retriever,
    ) = build_retriever(tmp_path)

    try:
        reranked_retriever.search(
            query="test",
            candidate_limit=2,
            result_limit=3,
        )
    except ValueError as exc:
        assert "cannot exceed" in str(exc)
    else:
        raise AssertionError(
            "Invalid candidate limits were accepted"
        )