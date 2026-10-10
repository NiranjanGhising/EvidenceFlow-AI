from pathlib import Path

from evidenceflow.core.config import Settings
from evidenceflow.domain.models import ProcessingStatus
from evidenceflow.ingestion.service import IngestionService
from evidenceflow.storage.documents import DocumentRepository


def create_service(
    tmp_path: Path,
) -> tuple[IngestionService, DocumentRepository]:
    settings = Settings.from_env(tmp_path)
    repository = DocumentRepository(settings.database_path)

    return IngestionService(settings, repository), repository


def test_document_create_duplicate_and_update(tmp_path: Path) -> None:
    service, repository = create_service(tmp_path)

    document_path = tmp_path / "guide.md"
    document_path.write_text(
        "# Guide\n\nVersion one.",
        encoding="utf-8",
    )

    created = service.ingest(document_path)

    assert created.outcome == "created"
    assert created.changed is False
    assert created.document.status is ProcessingStatus.PROCESSED
    assert repository.count() == 1
    assert repository.get_text(
        created.document.document_id
    ) == "# Guide\n\nVersion one."

    duplicate = service.ingest(document_path)

    assert duplicate.outcome == "duplicate"
    assert duplicate.changed is False
    assert duplicate.document.status is ProcessingStatus.SKIPPED_DUPLICATE
    assert repository.count() == 1

    document_path.write_text(
        "# Guide\n\nVersion two.",
        encoding="utf-8",
    )

    updated = service.ingest(document_path)

    assert updated.outcome == "updated"
    assert updated.changed is True
    assert updated.document.status is ProcessingStatus.PROCESSED
    assert repository.count() == 1
    assert repository.get_text(
        updated.document.document_id
    ) == "# Guide\n\nVersion two."


def test_same_content_with_different_name_is_duplicate(
    tmp_path: Path,
) -> None:
    service, repository = create_service(tmp_path)

    first_path = tmp_path / "first.txt"
    second_path = tmp_path / "second.txt"

    first_path.write_text("Identical content.", encoding="utf-8")
    second_path.write_text("Identical content.", encoding="utf-8")

    service.ingest(first_path)
    result = service.ingest(second_path)

    assert result.outcome == "duplicate"
    assert repository.count() == 1