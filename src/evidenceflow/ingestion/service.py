"""Document ingestion orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from evidenceflow.core.config import Settings
from evidenceflow.domain.models import Document, ProcessingStatus
from evidenceflow.ingestion.extractors import extract_document
from evidenceflow.ingestion.fingerprints import (
    document_id_for_path,
    sha256_file,
)
from evidenceflow.ingestion.validation import validate_file
from evidenceflow.storage.documents import DocumentRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Final result returned by the ingestion service."""

    document: Document
    outcome: str
    changed: bool
    page_count: int | None = None


class IngestionService:
    """Validate, extract, fingerprint, and persist a document."""

    def __init__(
        self,
        settings: Settings,
        repository: DocumentRepository | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository or DocumentRepository(
            settings.database_path
        )
        self.repository.initialize()

    def ingest(self, path: Path) -> IngestionResult:
        """Ingest one PDF, Markdown, or plain-text document."""

        validated = validate_file(
            path=path,
            max_upload_mb=self.settings.max_upload_mb,
        )

        content_hash = sha256_file(validated.path)

        duplicate = self.repository.find_by_hash(content_hash)

        if duplicate is not None:
            logger.info(
                "document_duplicate",
                extra={
                    "source_name": validated.source_name,
                    "document_id": duplicate.document_id,
                },
            )

            skipped_document = Document(
                document_id=duplicate.document_id,
                source_name=validated.source_name,
                source_type=validated.source_type,
                content_hash=content_hash,
                size_bytes=validated.size_bytes,
                status=ProcessingStatus.SKIPPED_DUPLICATE,
                metadata={
                    "duplicate_of": duplicate.document_id,
                },
            )

            return IngestionResult(
                document=skipped_document,
                outcome="duplicate",
                changed=False,
            )

        previous_document = self.repository.find_by_source_name(
            validated.source_name
        )

        extracted = extract_document(
            path=validated.path,
            source_type=validated.source_type,
        )

        document = Document(
            document_id=document_id_for_path(validated.path),
            source_name=validated.source_name,
            source_type=validated.source_type,
            content_hash=content_hash,
            size_bytes=validated.size_bytes,
            status=ProcessingStatus.PROCESSED,
            metadata={
                **extracted.metadata,
                "ingested_at": datetime.now(UTC).isoformat(),
            },
        )

        self.repository.upsert(
            document=document,
            extracted_text=extracted.text,
            page_count=extracted.page_count,
        )

        changed = (
            previous_document is not None
            and previous_document.content_hash != content_hash
        )

        outcome = "updated" if changed else "created"

        logger.info(
            "document_ingested",
            extra={
                "source_name": document.source_name,
                "document_id": document.document_id,
                "outcome": outcome,
            },
        )

        return IngestionResult(
            document=document,
            outcome=outcome,
            changed=changed,
            page_count=extracted.page_count,
        )