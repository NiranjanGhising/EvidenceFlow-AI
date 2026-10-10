"""SQLite persistence for document chunks."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from evidenceflow.domain.models import Chunk


class ChunkRepository:
    """Persist and retrieve chunk records."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row

        return connection

    def initialize(self) -> None:
        """Create the chunks table and indexes."""

        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    chunk_hash TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    chunk_order INTEGER NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    word_count INTEGER NOT NULL,
                    token_count INTEGER NOT NULL,
                    chunk_strategy TEXT NOT NULL,
                    page_number INTEGER,
                    heading_path_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(document_id)
                        REFERENCES documents(document_id)
                        ON DELETE CASCADE
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                ON chunks(document_id)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_hash
                ON chunks(chunk_hash)
                """
            )

            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_document_order
                ON chunks(document_id, chunk_order)
                """
            )

    def replace_for_document(
        self,
        document_id: str,
        chunks: list[Chunk],
    ) -> None:
        """Atomically replace all chunks belonging to a document."""

        with self._connect() as connection:
            connection.execute(
                "DELETE FROM chunks WHERE document_id = ?",
                (document_id,),
            )

            connection.executemany(
                """
                INSERT INTO chunks (
                    chunk_id,
                    chunk_hash,
                    document_id,
                    source_name,
                    source_type,
                    chunk_order,
                    start_char,
                    end_char,
                    text,
                    word_count,
                    token_count,
                    chunk_strategy,
                    page_number,
                    heading_path_json,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.chunk_hash,
                        chunk.document_id,
                        chunk.source_name,
                        chunk.source_type,
                        chunk.chunk_order,
                        chunk.start_char,
                        chunk.end_char,
                        chunk.text,
                        chunk.word_count,
                        chunk.token_count,
                        chunk.chunk_strategy,
                        chunk.page_number,
                        json.dumps(
                            list(chunk.heading_path),
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            chunk.metadata,
                            ensure_ascii=False,
                        ),
                    )
                    for chunk in chunks
                ],
            )

    def list_by_document(self, document_id: str) -> list[Chunk]:
        """Return chunks in their original document order."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM chunks
                WHERE document_id = ?
                ORDER BY chunk_order
                """,
                (document_id,),
            ).fetchall()

        return [self._to_chunk(row) for row in rows]

    def count_for_document(self, document_id: str) -> int:
        """Return the number of chunks stored for a document."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS chunk_count
                FROM chunks
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()

        return int(row["chunk_count"])

    def get(self, chunk_id: str) -> Chunk | None:
        """Return a single chunk by ID."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM chunks
                WHERE chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()

        return self._to_chunk(row) if row else None
    
    def list_all(self) -> list[Chunk]:
        """Return every persisted chunk in deterministic order."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM chunks
                ORDER BY document_id, chunk_order, chunk_id
                """
            ).fetchall()

        return [
            self._to_chunk(row)
            for row in rows
        ]

    @staticmethod
    def _to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"],
            chunk_hash=row["chunk_hash"],
            document_id=row["document_id"],
            source_name=row["source_name"],
            source_type=row["source_type"],
            chunk_order=row["chunk_order"],
            start_char=row["start_char"],
            end_char=row["end_char"],
            text=row["text"],
            word_count=row["word_count"],
            token_count=row["token_count"],
            chunk_strategy=row["chunk_strategy"],
            page_number=row["page_number"],
            heading_path=tuple(
                json.loads(row["heading_path_json"])
            ),
            metadata=json.loads(row["metadata_json"]),
        )