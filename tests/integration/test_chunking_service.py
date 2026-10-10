from pathlib import Path

from evidenceflow.chunking.service import ChunkingService
from evidenceflow.core.config import Settings
from evidenceflow.ingestion.service import IngestionService
from evidenceflow.storage.chunks import ChunkRepository
from evidenceflow.storage.documents import DocumentRepository


def create_services(
    tmp_path: Path,
) -> tuple[
    IngestionService,
    ChunkingService,
    DocumentRepository,
    ChunkRepository,
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
        chunk_size=120,
        overlap=20,
    )

    return (
        ingestion_service,
        chunking_service,
        document_repository,
        chunk_repository,
    )


def test_markdown_document_is_chunked_and_persisted(
    tmp_path: Path,
) -> None:
    (
        ingestion_service,
        chunking_service,
        _,
        chunk_repository,
    ) = create_services(tmp_path)

    document_path = tmp_path / "guide.md"

    document_path.write_text(
        "# FastAPI\n"
        "FastAPI validates incoming requests using models. "
        "The framework provides automatic API documentation.\n\n"
        "## Validation\n"
        "Pydantic validates request fields and reports errors. "
        "Validation prevents invalid values from reaching services.\n",
        encoding="utf-8",
    )

    ingestion_result = ingestion_service.ingest(
        document_path
    )

    result = chunking_service.chunk_document(
        ingestion_result.document.document_id
    )

    stored_chunks = chunk_repository.list_by_document(
        ingestion_result.document.document_id
    )

    assert result.chunk_count >= 2
    assert len(stored_chunks) == result.chunk_count
    assert stored_chunks[0].chunk_order == 0
    assert stored_chunks[0].chunk_strategy == (
        "markdown-recursive"
    )
    assert stored_chunks[0].heading_path == ("FastAPI",)

    for index, chunk in enumerate(stored_chunks):
        assert chunk.chunk_order == index
        assert chunk.chunk_id.startswith("chunk-")
        assert chunk.word_count > 0
        assert chunk.token_count > 0


def test_rechunking_replaces_previous_chunks(
    tmp_path: Path,
) -> None:
    (
        ingestion_service,
        chunking_service,
        _,
        chunk_repository,
    ) = create_services(tmp_path)

    document_path = tmp_path / "document.txt"

    document_path.write_text(
        " ".join(f"technical-word-{index}" for index in range(80)),
        encoding="utf-8",
    )

    ingestion_result = ingestion_service.ingest(
        document_path
    )

    first_result = chunking_service.chunk_document(
        ingestion_result.document.document_id
    )

    second_result = chunking_service.chunk_document(
        ingestion_result.document.document_id
    )

    assert first_result.chunk_count == second_result.chunk_count
    assert chunk_repository.count_for_document(
        ingestion_result.document.document_id
    ) == second_result.chunk_count

    assert [
        chunk.chunk_id for chunk in first_result.chunks
    ] == [
        chunk.chunk_id for chunk in second_result.chunks
    ]


def test_changed_document_gets_new_chunks(
    tmp_path: Path,
) -> None:
    (
        ingestion_service,
        chunking_service,
        _,
        chunk_repository,
    ) = create_services(tmp_path)

    document_path = tmp_path / "changing.md"

    document_path.write_text(
        "# Version One\nOriginal document content.",
        encoding="utf-8",
    )

    first_ingestion = ingestion_service.ingest(
        document_path
    )

    first_chunks = chunking_service.chunk_document(
        first_ingestion.document.document_id
    )

    first_ids = {
        chunk.chunk_id for chunk in first_chunks.chunks
    }

    document_path.write_text(
        "# Version Two\nUpdated document content with new evidence.",
        encoding="utf-8",
    )

    second_ingestion = ingestion_service.ingest(
        document_path
    )

    second_chunks = chunking_service.chunk_document(
        second_ingestion.document.document_id
    )

    second_ids = {
        chunk.chunk_id for chunk in second_chunks.chunks
    }

    assert second_ingestion.outcome == "updated"
    assert first_ids != second_ids
    assert chunk_repository.count_for_document(
        second_ingestion.document.document_id
    ) == second_chunks.chunk_count