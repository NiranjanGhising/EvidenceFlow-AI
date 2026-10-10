from pathlib import Path

import numpy as np

from evidenceflow.chunking.service import ChunkingService
from evidenceflow.core.config import Settings
from evidenceflow.indexing.service import EmbeddingIndexService
from evidenceflow.ingestion.service import IngestionService
from evidenceflow.retrieval.service import HybridRetriever
from evidenceflow.retrieval.sparse import BM25Retriever
from evidenceflow.storage.chunks import ChunkRepository
from evidenceflow.storage.documents import DocumentRepository
from evidenceflow.storage.vectors import VectorRepository


class FakeEncoder:
    @property
    def model_name(self) -> str:
        return "hybrid-test-encoder"

    @property
    def dimension(self) -> int:
        return 4

    def encode(self, texts: list[str]) -> np.ndarray:
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
                        or "security" in lowered
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


def build_hybrid_retriever(
    tmp_path: Path,
) -> tuple[
    IngestionService,
    ChunkingService,
    EmbeddingIndexService,
    HybridRetriever,
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

    return (
        ingestion_service,
        chunking_service,
        indexing_service,
        hybrid_retriever,
    )


def ingest_chunk_and_index(
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


def test_hybrid_search_returns_matching_document(
    tmp_path: Path,
) -> None:
    (
        ingestion_service,
        chunking_service,
        indexing_service,
        hybrid_retriever,
    ) = build_hybrid_retriever(tmp_path)

    ingest_chunk_and_index(
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

    ingest_chunk_and_index(
        ingestion_service,
        chunking_service,
        indexing_service,
        tmp_path / "database.md",
        (
            "# Database\n"
            "PostgreSQL stores persistent application records."
        ),
    )

    response = hybrid_retriever.search(
        "How does FastAPI validate request bodies?",
        candidate_limit=10,
        result_limit=5,
    )

    assert response.results
    assert response.results[0].source_name == "api.md"
    assert response.results[0].fusion_rank == 1
    assert "FastAPI" in response.results[0].text


def test_hybrid_results_have_score_provenance(
    tmp_path: Path,
) -> None:
    (
        ingestion_service,
        chunking_service,
        indexing_service,
        hybrid_retriever,
    ) = build_hybrid_retriever(tmp_path)

    ingest_chunk_and_index(
        ingestion_service,
        chunking_service,
        indexing_service,
        tmp_path / "security.md",
        (
            "# Authorization\n"
            "Authorization protects restricted resources."
        ),
    )

    response = hybrid_retriever.search(
        "authorization restricted resources",
        candidate_limit=5,
        result_limit=5,
    )

    result = response.results[0]

    assert result.sparse_rank is not None
    assert result.dense_rank is not None
    assert result.sparse_score is not None
    assert result.dense_score is not None
    assert result.fusion_score > 0


def test_hybrid_search_reports_latency(
    tmp_path: Path,
) -> None:
    (
        ingestion_service,
        chunking_service,
        indexing_service,
        hybrid_retriever,
    ) = build_hybrid_retriever(tmp_path)

    ingest_chunk_and_index(
        ingestion_service,
        chunking_service,
        indexing_service,
        tmp_path / "guide.txt",
        "FastAPI validates incoming request data.",
    )

    response = hybrid_retriever.search(
        "FastAPI request validation"
    )

    assert set(response.latency_ms) == {
        "sparse",
        "dense",
        "fusion",
        "total",
    }

    assert response.latency_ms["total"] >= 0


def test_hybrid_search_rejects_empty_query(
    tmp_path: Path,
) -> None:
    (
        _,
        _,
        _,
        hybrid_retriever,
    ) = build_hybrid_retriever(tmp_path)

    try:
        hybrid_retriever.search("   ")
    except ValueError as exc:
        assert "cannot be empty" in str(exc)
    else:
        raise AssertionError(
            "Empty hybrid query was not rejected"
        )