from pathlib import Path

import numpy as np

from evidenceflow.chunking.service import ChunkingService
from evidenceflow.core.config import Settings
from evidenceflow.indexing.service import EmbeddingIndexService
from evidenceflow.ingestion.service import IngestionService
from evidenceflow.storage.chunks import ChunkRepository
from evidenceflow.storage.documents import DocumentRepository
from evidenceflow.storage.vectors import VectorRepository


class FakeEncoder:
    """Deterministic test encoder without model downloads."""

    @property
    def model_name(self) -> str:
        return "test-encoder-v1"

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
                        "security" in lowered
                        or "authorization" in lowered
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


def build_services(
    tmp_path: Path,
) -> tuple[
    IngestionService,
    ChunkingService,
    EmbeddingIndexService,
    VectorRepository,
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
        chunk_size=100,
        overlap=15,
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

    return (
        ingestion_service,
        chunking_service,
        indexing_service,
        vector_repository,
    )


def test_document_chunks_are_indexed(
    tmp_path: Path,
) -> None:
    (
        ingestion_service,
        chunking_service,
        indexing_service,
        vector_repository,
    ) = build_services(tmp_path)

    document_path = tmp_path / "fastapi.md"

    document_path.write_text(
        "# FastAPI\n"
        "FastAPI validates requests using Pydantic models.\n\n"
        "## Database\n"
        "SQL stores persistent application data.",
        encoding="utf-8",
    )

    ingestion_result = ingestion_service.ingest(
        document_path
    )

    chunking_result = chunking_service.chunk_document(
        ingestion_result.document.document_id
    )

    indexing_result = indexing_service.index_document(
        ingestion_result.document.document_id
    )

    assert indexing_result.vector_count == (
        chunking_result.chunk_count
    )

    assert indexing_result.dimension == 4

    assert vector_repository.count(
        model_name="test-encoder-v1",
        document_id=ingestion_result.document.document_id,
    ) == chunking_result.chunk_count


def test_semantic_search_returns_relevant_chunk(
    tmp_path: Path,
) -> None:
    (
        ingestion_service,
        chunking_service,
        indexing_service,
        _,
    ) = build_services(tmp_path)

    document_path = tmp_path / "guide.md"

    document_path.write_text(
        "# API\n"
        "FastAPI validates incoming request bodies.\n\n"
        "# Storage\n"
        "SQL databases persist application records.\n\n"
        "# Security\n"
        "Authorization protects restricted resources.",
        encoding="utf-8",
    )

    ingestion_result = ingestion_service.ingest(
        document_path
    )

    chunking_service.chunk_document(
        ingestion_result.document.document_id
    )

    indexing_service.index_document(
        ingestion_result.document.document_id
    )

    results = indexing_service.search(
        "How are API requests validated?",
        limit=3,
    )

    assert results
    assert results[0].source_name == "guide.md"
    assert "FastAPI" in results[0].text


def test_reindexing_replaces_old_vectors(
    tmp_path: Path,
) -> None:
    (
        ingestion_service,
        chunking_service,
        indexing_service,
        vector_repository,
    ) = build_services(tmp_path)

    document_path = tmp_path / "guide.txt"

    document_path.write_text(
        "FastAPI validates request data.",
        encoding="utf-8",
    )

    ingestion_result = ingestion_service.ingest(
        document_path
    )

    chunking_service.chunk_document(
        ingestion_result.document.document_id
    )

    first_result = indexing_service.index_document(
        ingestion_result.document.document_id
    )

    second_result = indexing_service.index_document(
        ingestion_result.document.document_id
    )

    assert first_result.vector_count == second_result.vector_count

    assert vector_repository.count(
        model_name="test-encoder-v1",
        document_id=ingestion_result.document.document_id,
    ) == second_result.vector_count


def test_search_rejects_empty_query(
    tmp_path: Path,
) -> None:
    (
        _,
        _,
        indexing_service,
        _,
    ) = build_services(tmp_path)

    try:
        indexing_service.search("   ")
    except ValueError as exc:
        assert "cannot be empty" in str(exc)
    else:
        raise AssertionError(
            "Empty search query was not rejected"
        )