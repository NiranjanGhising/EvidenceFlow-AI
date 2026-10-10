"""Chunk embedding and semantic-search services."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from evidenceflow.embeddings.encoder import EmbeddingEncoder
from evidenceflow.embeddings.models import SemanticSearchResult
from evidenceflow.storage.chunks import ChunkRepository
from evidenceflow.storage.documents import DocumentRepository
from evidenceflow.storage.vectors import (
    VectorRecord,
    VectorRepository,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IndexingResult:
    """Summary returned after indexing one document."""

    document_id: str
    source_name: str
    model_name: str
    vector_count: int
    dimension: int


class EmbeddingIndexService:
    """Generate, persist, replace, and search chunk vectors."""

    def __init__(
        self,
        document_repository: DocumentRepository,
        chunk_repository: ChunkRepository,
        vector_repository: VectorRepository,
        encoder: EmbeddingEncoder,
    ) -> None:
        self.document_repository = document_repository
        self.chunk_repository = chunk_repository
        self.vector_repository = vector_repository
        self.encoder = encoder

        self.vector_repository.initialize()

    def index_document(self, document_id: str) -> IndexingResult:
        """Generate and persist embeddings for one chunked document."""

        document = self.document_repository.find_by_id(
            document_id
        )

        if document is None:
            raise ValueError(
                f"Document was not found: {document_id}"
            )

        chunks = self.chunk_repository.list_by_document(
            document_id
        )

        if not chunks:
            raise ValueError(
                "Document contains no chunks. "
                "Run evidenceflow-chunk before indexing."
            )

        vectors = self.encoder.encode(
            [chunk.text for chunk in chunks]
        )

        if vectors.shape[0] != len(chunks):
            raise RuntimeError(
                "Generated vector count does not match chunk count"
            )

        records = [
            VectorRecord(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                model_name=self.encoder.model_name,
                vector=np.asarray(
                    vectors[index],
                    dtype=np.float32,
                ),
            )
            for index, chunk in enumerate(chunks)
        ]

        self.vector_repository.replace_document_vectors(
            document_id=document_id,
            model_name=self.encoder.model_name,
            records=records,
        )

        logger.info(
            "document_vectors_indexed",
            extra={
                "document_id": document_id,
                "source_name": document.source_name,
                "model_name": self.encoder.model_name,
                "vector_count": len(records),
                "dimension": int(vectors.shape[1]),
            },
        )

        return IndexingResult(
            document_id=document_id,
            source_name=document.source_name,
            model_name=self.encoder.model_name,
            vector_count=len(records),
            dimension=int(vectors.shape[1]),
        )

    def search(
        self,
        query: str,
        limit: int = 5,
        minimum_score: float | None = None,
        document_id: str | None = None,
    ) -> list[SemanticSearchResult]:
        """Search indexed chunks using semantic similarity."""

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "Search query cannot be empty"
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero"
            )

        normalized_document_id = None

        if document_id is not None:
            normalized_document_id = (
                document_id.strip()
            )

            if not normalized_document_id:
                raise ValueError(
                    "document_id cannot be empty"
                )

        query_vectors = self.encoder.encode(
            [normalized_query]
        )

        return self.vector_repository.search(
            query_vector=query_vectors[0],
            model_name=self.encoder.model_name,
            limit=limit,
            minimum_score=minimum_score,
            document_id=normalized_document_id,
        )