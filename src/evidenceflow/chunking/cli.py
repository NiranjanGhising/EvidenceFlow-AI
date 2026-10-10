"""Command-line interface for chunking an ingested document."""

from __future__ import annotations

import argparse
import json

from evidenceflow.chunking.service import ChunkingService
from evidenceflow.core.config import Settings
from evidenceflow.core.logging import configure_logging
from evidenceflow.storage.chunks import ChunkRepository
from evidenceflow.storage.documents import DocumentRepository


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk an ingested EvidenceFlow document"
    )

    parser.add_argument(
        "document_id",
        help="Document ID returned by evidenceflow-ingest",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Maximum chunk size in characters",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=120,
        help="Approximate overlap in characters",
    )

    arguments = parser.parse_args()

    settings = Settings.from_env()
    settings.ensure_directories()
    configure_logging(settings.log_level)

    document_repository = DocumentRepository(settings.database_path)
    document_repository.initialize()

    chunk_repository = ChunkRepository(settings.database_path)

    service = ChunkingService(
        document_repository=document_repository,
        chunk_repository=chunk_repository,
        chunk_size=arguments.chunk_size,
        overlap=arguments.overlap,
    )

    result = service.chunk_document(arguments.document_id)

    print(
        json.dumps(
            {
                "document_id": result.document_id,
                "source_name": result.source_name,
                "strategy": result.strategy,
                "chunk_count": result.chunk_count,
                "chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "chunk_order": chunk.chunk_order,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "word_count": chunk.word_count,
                        "token_count": chunk.token_count,
                        "heading_path": list(chunk.heading_path),
                    }
                    for chunk in result.chunks
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()