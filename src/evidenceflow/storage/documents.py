"""SQLite document metadata repository."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from evidenceflow.domain.models import Document, ProcessingStatus


class DocumentRepository:
    """Persist document metadata and extracted text in SQLite."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row

        return connection

    def initialize(self) -> None:
        """Create the document table and required indexes."""

        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    source_name TEXT NOT NULL UNIQUE,
                    source_type TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    extracted_text TEXT,
                    page_count INTEGER,
                    metadata_json TEXT NOT NULL,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_content_hash
                ON documents(content_hash)
                """
            )

    def find_by_hash(self, content_hash: str) -> Document | None:
        """Find an existing document using its content fingerprint."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM documents
                WHERE content_hash = ?
                LIMIT 1
                """,
                (content_hash,),
            ).fetchone()

        return self._to_document(row) if row else None

    def find_by_source_name(self, source_name: str) -> Document | None:
        """Find a document using its source filename."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM documents
                WHERE source_name = ?
                LIMIT 1
                """,
                (source_name,),
            ).fetchone()

        return self._to_document(row) if row else None

    def upsert(
        self,
        document: Document,
        extracted_text: str | None,
        page_count: int | None,
        error_message: str | None = None,
    ) -> None:
        """Insert a document or replace the record with the same filename."""

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id,
                    source_name,
                    source_type,
                    content_hash,
                    size_bytes,
                    status,
                    extracted_text,
                    page_count,
                    metadata_json,
                    error_message,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(source_name) DO UPDATE SET
                    document_id = excluded.document_id,
                    source_type = excluded.source_type,
                    content_hash = excluded.content_hash,
                    size_bytes = excluded.size_bytes,
                    status = excluded.status,
                    extracted_text = excluded.extracted_text,
                    page_count = excluded.page_count,
                    metadata_json = excluded.metadata_json,
                    error_message = excluded.error_message,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    document.document_id,
                    document.source_name,
                    document.source_type,
                    document.content_hash,
                    document.size_bytes,
                    document.status.value,
                    extracted_text,
                    page_count,
                    json.dumps(document.metadata, ensure_ascii=False),
                    error_message,
                    document.created_at.isoformat(),
                ),
            )

    def count(self) -> int:
        """Return the number of stored document records."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS document_count FROM documents"
            ).fetchone()

        return int(row["document_count"])

    def get_text(self, document_id: str) -> str | None:
        """Return persisted extracted text."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT extracted_text
                FROM documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()

        if row is None:
            return None

        return row["extracted_text"]

    def find_by_id(self, document_id: str) -> Document | None:
        """Find a document using its stable document ID."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM documents
                WHERE document_id = ?
                LIMIT 1
                """,
                (document_id,),
            ).fetchone()

        return self._to_document(row) if row else None

    @staticmethod
    def _to_document(row: sqlite3.Row) -> Document:
        return Document(
            document_id=row["document_id"],
            source_name=row["source_name"],
            source_type=row["source_type"],
            content_hash=row["content_hash"],
            size_bytes=row["size_bytes"],
            status=ProcessingStatus(row["status"]),
            metadata=json.loads(row["metadata_json"]),
        )