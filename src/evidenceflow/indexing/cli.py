"""Commands for embedding indexing and semantic search."""

from __future__ import annotations

import argparse
import json

from evidenceflow.core.config import Settings
from evidenceflow.core.logging import configure_logging
from evidenceflow.embeddings.encoder import (
    SentenceTransformerEncoder,
)
from evidenceflow.indexing.service import EmbeddingIndexService
from evidenceflow.storage.chunks import ChunkRepository
from evidenceflow.storage.documents import DocumentRepository
from evidenceflow.storage.vectors import VectorRepository


def build_service(
    settings: Settings,
) -> EmbeddingIndexService:
    """Construct the indexing service from application settings."""

    document_repository = DocumentRepository(
        settings.database_path
    )
    document_repository.initialize()

    chunk_repository = ChunkRepository(
        settings.database_path
    )
    chunk_repository.initialize()

    vector_repository = VectorRepository(
        settings.database_path
    )

    encoder = SentenceTransformerEncoder(
        model_name=settings.embedding_model,
        device="cpu",
    )

    return EmbeddingIndexService(
        document_repository=document_repository,
        chunk_repository=chunk_repository,
        vector_repository=vector_repository,
        encoder=encoder,
    )


def index_command(document_id: str) -> None:
    settings = Settings.from_env()
    settings.ensure_directories()
    configure_logging(settings.log_level)

    result = build_service(settings).index_document(
        document_id
    )

    print(
        json.dumps(
            {
                "document_id": result.document_id,
                "source_name": result.source_name,
                "model_name": result.model_name,
                "vector_count": result.vector_count,
                "dimension": result.dimension,
            },
            indent=2,
        )
    )


def search_command(
    query: str,
    limit: int,
    minimum_score: float | None,
) -> None:
    settings = Settings.from_env()
    settings.ensure_directories()
    configure_logging(settings.log_level)

    results = build_service(settings).search(
        query=query,
        limit=limit,
        minimum_score=minimum_score,
    )

    print(
        json.dumps(
            {
                "query": query,
                "model_name": settings.embedding_model,
                "result_count": len(results),
                "results": [
                    {
                        "rank": index,
                        "chunk_id": result.chunk_id,
                        "document_id": result.document_id,
                        "source_name": result.source_name,
                        "score": round(result.score, 6),
                        "chunk_order": result.chunk_order,
                        "heading_path": list(
                            result.heading_path
                        ),
                        "text": result.text,
                    }
                    for index, result in enumerate(
                        results,
                        start=1,
                    )
                ],
            },
            indent=2,
        )
    )


def index_main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate embeddings for a chunked document"
    )

    parser.add_argument(
        "document_id",
        help="Document ID returned by evidenceflow-ingest",
    )

    arguments = parser.parse_args()
    index_command(arguments.document_id)


def search_main() -> None:
    parser = argparse.ArgumentParser(
        description="Search EvidenceFlow using semantic similarity"
    )

    parser.add_argument(
        "query",
        help="Natural-language search query",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results",
    )

    parser.add_argument(
        "--minimum-score",
        type=float,
        default=None,
        help="Optional minimum cosine similarity",
    )

    arguments = parser.parse_args()

    search_command(
        query=arguments.query,
        limit=arguments.limit,
        minimum_score=arguments.minimum_score,
    )


if __name__ == "__main__":
    index_main()