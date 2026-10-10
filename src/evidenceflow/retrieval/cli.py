"""Command-line hybrid retrieval."""

from __future__ import annotations

import argparse
import json

from evidenceflow.core.config import Settings
from evidenceflow.core.logging import configure_logging
from evidenceflow.indexing.cli import build_service
from evidenceflow.retrieval.service import HybridRetriever
from evidenceflow.retrieval.sparse import BM25Retriever
from evidenceflow.storage.chunks import ChunkRepository


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search EvidenceFlow using hybrid retrieval"
    )

    parser.add_argument(
        "query",
        help="Natural-language or keyword search query",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of fused results",
    )

    parser.add_argument(
        "--candidates",
        type=int,
        default=20,
        help="Candidate count from each retriever",
    )

    parser.add_argument(
        "--sparse-weight",
        type=float,
        default=1.0,
        help="BM25 contribution to RRF",
    )

    parser.add_argument(
        "--dense-weight",
        type=float,
        default=1.0,
        help="Dense retrieval contribution to RRF",
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="Reciprocal Rank Fusion constant",
    )

    arguments = parser.parse_args()

    settings = Settings.from_env()
    settings.ensure_directories()
    configure_logging(settings.log_level)

    chunk_repository = ChunkRepository(
        settings.database_path
    )
    chunk_repository.initialize()

    sparse_retriever = BM25Retriever(
        chunk_repository=chunk_repository,
    )

    dense_retriever = build_service(settings)

    hybrid_retriever = HybridRetriever(
        sparse_retriever=sparse_retriever,
        dense_retriever=dense_retriever,
        sparse_weight=arguments.sparse_weight,
        dense_weight=arguments.dense_weight,
        rrf_k=arguments.rrf_k,
    )

    response = hybrid_retriever.search(
        query=arguments.query,
        candidate_limit=arguments.candidates,
        result_limit=arguments.limit,
    )

    print(
        json.dumps(
            {
                "query": response.query,
                "retrieval_mode": "hybrid_rrf",
                "sparse_candidate_count": len(
                    response.sparse_results
                ),
                "dense_candidate_count": len(
                    response.dense_results
                ),
                "result_count": len(response.results),
                "latency_ms": {
                    name: round(value, 3)
                    for name, value in (
                        response.latency_ms.items()
                    )
                },
                "results": [
                    {
                        "rank": result.fusion_rank,
                        "chunk_id": result.chunk_id,
                        "document_id": result.document_id,
                        "source_name": result.source_name,
                        "fusion_score": round(
                            result.fusion_score,
                            8,
                        ),
                        "sparse_score": (
                            round(result.sparse_score, 6)
                            if result.sparse_score is not None
                            else None
                        ),
                        "sparse_rank": result.sparse_rank,
                        "dense_score": (
                            round(result.dense_score, 6)
                            if result.dense_score is not None
                            else None
                        ),
                        "dense_rank": result.dense_rank,
                        "chunk_order": result.chunk_order,
                        "heading_path": list(
                            result.heading_path
                        ),
                        "text": result.text,
                    }
                    for result in response.results
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
