"""Command-line hybrid retrieval with cross-encoder reranking."""

from __future__ import annotations

import argparse
import json

from evidenceflow.core.config import Settings
from evidenceflow.core.logging import configure_logging
from evidenceflow.indexing.cli import build_service
from evidenceflow.reranking.encoder import CrossEncoderReranker
from evidenceflow.reranking.service import (
    RerankedRetriever,
    RerankingService,
)
from evidenceflow.retrieval.service import HybridRetriever
from evidenceflow.retrieval.sparse import BM25Retriever
from evidenceflow.storage.chunks import ChunkRepository


def main() -> None:
    """Run hybrid retrieval followed by cross-encoder reranking."""

    parser = argparse.ArgumentParser(
        description=(
            "Search EvidenceFlow using hybrid retrieval "
            "and cross-encoder reranking"
        )
    )

    parser.add_argument(
        "query",
        help="Natural-language or keyword search query",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of reranked results",
    )

    parser.add_argument(
        "--candidates",
        type=int,
        default=20,
        help="Hybrid candidates sent to the reranker",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Cross-encoder prediction batch size",
    )

    parser.add_argument(
        "--sparse-weight",
        type=float,
        default=1.0,
        help="BM25 contribution to weighted RRF",
    )

    parser.add_argument(
        "--dense-weight",
        type=float,
        default=1.0,
        help="Dense contribution to weighted RRF",
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="Reciprocal Rank Fusion constant",
    )

    parser.add_argument(
        "--disable-fallback",
        action="store_true",
        help="Raise reranker errors instead of using fusion order",
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

    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model,
        batch_size=arguments.batch_size,
        device="cpu",
    )

    reranking_service = RerankingService(
        reranker=reranker,
        fallback_enabled=not arguments.disable_fallback,
    )

    retriever = RerankedRetriever(
        hybrid_retriever=hybrid_retriever,
        reranking_service=reranking_service,
    )

    response = retriever.search(
        query=arguments.query,
        candidate_limit=arguments.candidates,
        result_limit=arguments.limit,
    )

    output = {
        "query": response.query,
        "retrieval_mode": "hybrid_rrf_cross_encoder",
        "reranker_model": response.model_name,
        "used_fallback": response.used_fallback,
        "fallback_reason": response.fallback_reason,
        "hybrid_candidate_count": len(
            response.hybrid_response.results
        ),
        "result_count": len(response.results),
        "latency_ms": {
            name: round(value, 3)
            for name, value in response.latency_ms.items()
        },
        "results": [
            {
                "rank": result.rerank_rank,
                "chunk_id": result.chunk_id,
                "document_id": result.document_id,
                "source_name": result.source_name,
                "rerank_score": round(
                    result.rerank_score,
                    6,
                ),
                "previous_fusion_rank": result.fusion_rank,
                "fusion_score": round(
                    result.fusion_score,
                    8,
                ),
                "sparse_rank": result.sparse_rank,
                "sparse_score": (
                    round(result.sparse_score, 6)
                    if result.sparse_score is not None
                    else None
                ),
                "dense_rank": result.dense_rank,
                "dense_score": (
                    round(result.dense_score, 6)
                    if result.dense_score is not None
                    else None
                ),
                "chunk_order": result.chunk_order,
                "heading_path": list(
                    result.heading_path
                ),
                "text": result.text,
            }
            for result in response.results
        ],
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()