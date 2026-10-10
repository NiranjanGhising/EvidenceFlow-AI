"""SQLite persistence and cosine search for embedding vectors."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from evidenceflow.embeddings.models import SemanticSearchResult


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """Vector data ready for persistence."""

    chunk_id: str
    document_id: str
    model_name: str
    vector: np.ndarray


class VectorRepository:
    """Persist normalized vectors and perform exact cosine search."""

    def __init__(
        self,
        database_path: Path,
    ) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _connect(
        self,
    ) -> sqlite3.Connection:
        """Return a configured SQLite connection."""

        connection = sqlite3.connect(
            self.database_path
        )

        connection.row_factory = sqlite3.Row

        return connection

    def initialize(
        self,
    ) -> None:
        """Create the embedding table and indexes."""

        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector_blob BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(chunk_id, model_name),
                    FOREIGN KEY(chunk_id)
                        REFERENCES chunks(chunk_id)
                        ON DELETE CASCADE,
                    FOREIGN KEY(document_id)
                        REFERENCES documents(document_id)
                        ON DELETE CASCADE
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_embeddings_document
                ON embeddings(document_id)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_embeddings_model
                ON embeddings(model_name)
                """
            )

    def replace_document_vectors(
        self,
        document_id: str,
        model_name: str,
        records: list[VectorRecord],
    ) -> None:
        """Atomically replace vectors for one document and model."""

        if any(
            record.document_id != document_id
            for record in records
        ):
            raise ValueError(
                "All vector records must belong to "
                "the same document"
            )

        if any(
            record.model_name != model_name
            for record in records
        ):
            raise ValueError(
                "All vector records must use the "
                "same embedding model"
            )

        timestamp = datetime.now(
            UTC
        ).isoformat()

        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM embeddings
                WHERE document_id = ?
                AND model_name = ?
                """,
                (
                    document_id,
                    model_name,
                ),
            )

            connection.executemany(
                """
                INSERT INTO embeddings (
                    chunk_id,
                    document_id,
                    model_name,
                    dimension,
                    vector_blob,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.chunk_id,
                        record.document_id,
                        record.model_name,
                        int(record.vector.size),
                        np.asarray(
                            record.vector,
                            dtype=np.float32,
                        ).tobytes(),
                        timestamp,
                        timestamp,
                    )
                    for record in records
                ],
            )

    def count(
        self,
        model_name: str | None = None,
        document_id: str | None = None,
    ) -> int:
        """Count stored embedding vectors."""

        conditions: list[str] = []
        parameters: list[str] = []

        if model_name is not None:
            normalized_model_name = (
                model_name.strip()
            )

            if not normalized_model_name:
                raise ValueError(
                    "model_name cannot be empty"
                )

            conditions.append(
                "model_name = ?"
            )

            parameters.append(
                normalized_model_name
            )

        if document_id is not None:
            normalized_document_id = (
                document_id.strip()
            )

            if not normalized_document_id:
                raise ValueError(
                    "document_id cannot be empty"
                )

            conditions.append(
                "document_id = ?"
            )

            parameters.append(
                normalized_document_id
            )

        where_clause = ""

        if conditions:
            where_clause = (
                " WHERE "
                + " AND ".join(conditions)
            )

        query = (
            "SELECT COUNT(*) AS vector_count "
            "FROM embeddings"
        )

        query += where_clause

        with self._connect() as connection:
            row = connection.execute(
                query,
                parameters,
            ).fetchone()

        return int(
            row["vector_count"]
        )

    def _load_search_rows(
        self,
        model_name: str,
        document_id: str | None,
    ) -> list[sqlite3.Row]:
        """Load vectors for a model and optional document."""

        query = """
            SELECT
                embeddings.chunk_id,
                embeddings.document_id,
                embeddings.dimension,
                embeddings.vector_blob,
                chunks.source_name,
                chunks.text,
                chunks.chunk_order,
                chunks.heading_path_json
            FROM embeddings
            INNER JOIN chunks
                ON chunks.chunk_id = embeddings.chunk_id
            WHERE embeddings.model_name = ?
        """

        parameters: list[str] = [
            model_name
        ]

        if document_id is not None:
            normalized_document_id = (
                document_id.strip()
            )

            if not normalized_document_id:
                raise ValueError(
                    "document_id cannot be empty"
                )

            query += (
                "\nAND embeddings.document_id = ?"
            )

            parameters.append(
                normalized_document_id
            )

        with self._connect() as connection:
            return connection.execute(
                query,
                parameters,
            ).fetchall()

    def search(
        self,
        query_vector: np.ndarray,
        model_name: str,
        limit: int = 5,
        minimum_score: float | None = None,
        document_id: str | None = None,
    ) -> list[SemanticSearchResult]:
        """Return document-scoped cosine-similarity results."""

        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero"
            )

        normalized_model_name = (
            model_name.strip()
        )

        if not normalized_model_name:
            raise ValueError(
                "model_name cannot be empty"
            )

        query_vector = np.asarray(
            query_vector,
            dtype=np.float32,
        )

        if query_vector.ndim != 1:
            raise ValueError(
                "query_vector must be one-dimensional"
            )

        query_norm = float(
            np.linalg.norm(query_vector)
        )

        if query_norm == 0:
            raise ValueError(
                "query_vector cannot be a zero vector"
            )

        normalized_query = (
            query_vector / query_norm
        )

        rows = self._load_search_rows(
            model_name=normalized_model_name,
            document_id=document_id,
        )

        if not rows:
            return []

        dimensions = {
            int(row["dimension"])
            for row in rows
        }

        if len(dimensions) != 1:
            raise RuntimeError(
                "Stored embeddings contain "
                "inconsistent dimensions"
            )

        stored_dimension = next(
            iter(dimensions)
        )

        if stored_dimension != query_vector.size:
            raise ValueError(
                "Query vector dimension does not "
                "match stored vectors"
            )

        vectors = np.vstack(
            [
                np.frombuffer(
                    row["vector_blob"],
                    dtype=np.float32,
                )
                for row in rows
            ]
        )

        vector_norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True,
        )

        vector_norms[
            vector_norms == 0
        ] = 1.0

        normalized_vectors = (
            vectors / vector_norms
        )

        scores = (
            normalized_vectors
            @ normalized_query
        )

        ranked_indices = np.argsort(
            -scores
        )

        results: list[
            SemanticSearchResult
        ] = []

        for index in ranked_indices:
            score = float(
                scores[index]
            )

            if (
                minimum_score is not None
                and score < minimum_score
            ):
                continue

            row = rows[
                int(index)
            ]

            results.append(
                SemanticSearchResult(
                    chunk_id=row["chunk_id"],
                    document_id=row[
                        "document_id"
                    ],
                    source_name=row[
                        "source_name"
                    ],
                    text=row["text"],
                    score=score,
                    chunk_order=row[
                        "chunk_order"
                    ],
                    heading_path=tuple(
                        json.loads(
                            row[
                                "heading_path_json"
                            ]
                        )
                    ),
                )
            )

            if len(results) >= limit:
                break

        return results