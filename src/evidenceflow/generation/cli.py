"""Command-line grounded question answering."""

from __future__ import annotations

import argparse
import json

from evidenceflow.core.config import Settings
from evidenceflow.core.logging import configure_logging
from evidenceflow.generation.service import (
    EvidenceSufficiencyChecker,
    ExtractiveAnswerGenerator,
    GroundedAnswerService,
)
from evidenceflow.indexing.cli import build_service
from evidenceflow.reranking.encoder import CrossEncoderReranker
from evidenceflow.reranking.service import (
    RerankedRetriever,
    RerankingService,
)
from evidenceflow.retrieval.service import HybridRetriever
from evidenceflow.retrieval.sparse import BM25Retriever
from evidenceflow.storage.chunks import ChunkRepository


def build_answer_service(
    settings: Settings,
    minimum_rerank_score: float,
) -> GroundedAnswerService:
    """Construct the complete grounded-answer pipeline."""

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
    )

    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model,
        device="cpu",
    )

    reranking_service = RerankingService(
        reranker=reranker,
        fallback_enabled=True,
    )

    reranked_retriever = RerankedRetriever(
        hybrid_retriever=hybrid_retriever,
        reranking_service=reranking_service,
    )

    sufficiency_checker = EvidenceSufficiencyChecker(
        minimum_rerank_score=minimum_rerank_score,
        maximum_evidence=2,
    )

    return GroundedAnswerService(
        retriever=reranked_retriever,
        sufficiency_checker=sufficiency_checker,
        answer_generator=ExtractiveAnswerGenerator(),
    )


def main() -> None:
    """Run grounded question answering."""

    parser = argparse.ArgumentParser(
        description=(
            "Answer a question using independently "
            "validated document citations"
        )
    )

    parser.add_argument(
        "question",
        help="Question to answer from indexed documents",
    )

    parser.add_argument(
        "--candidates",
        type=int,
        default=10,
        help="Hybrid candidates sent to the reranker",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum reranked evidence results",
    )

    parser.add_argument(
        "--minimum-rerank-score",
        type=float,
        default=0.0,
        help="Minimum cross-encoder score for answer support",
    )

    arguments = parser.parse_args()

    settings = Settings.from_env()
    settings.ensure_directories()
    configure_logging(settings.log_level)

    service = build_answer_service(
        settings=settings,
        minimum_rerank_score=(
            arguments.minimum_rerank_score
        ),
    )

    result = service.answer(
        question=arguments.question,
        candidate_limit=arguments.candidates,
        result_limit=arguments.limit,
    )

    output = {
        "question": result.question,
        "answer": result.answer,
        "supported": result.supported,
        "refusal_reason": result.refusal_reason,
        "retrieval_mode": result.retrieval_mode,
        "reranker_model": result.model_name,
        "used_reranker_fallback": (
            result.used_reranker_fallback
        ),
        "citation_validation": {
            "valid": result.citation_validation.valid,
            "coverage": round(
                result.citation_validation.coverage,
                4,
            ),
            "valid_citation_ids": list(
                result.citation_validation.valid_citation_ids
            ),
            "invalid_citation_ids": list(
                result.citation_validation.invalid_citation_ids
            ),
            "missing_citation_ids": list(
                result.citation_validation.missing_citation_ids
            ),
        },
        "citations": [
            {
                "citation_id": citation.citation_id,
                "chunk_id": citation.chunk_id,
                "document_id": citation.document_id,
                "source_name": citation.source_name,
                "heading_path": list(
                    citation.heading_path
                ),
            }
            for citation in result.citations
        ],
        "evidence": [
            {
                "chunk_id": evidence.chunk_id,
                "document_id": evidence.document_id,
                "source_name": evidence.source_name,
                "rerank_score": round(
                    evidence.rerank_score,
                    6,
                ),
                "rerank_rank": evidence.rerank_rank,
                "fusion_score": round(
                    evidence.fusion_score,
                    8,
                ),
                "fusion_rank": evidence.fusion_rank,
                "sparse_score": (
                    round(evidence.sparse_score, 6)
                    if evidence.sparse_score is not None
                    else None
                ),
                "sparse_rank": evidence.sparse_rank,
                "dense_score": (
                    round(evidence.dense_score, 6)
                    if evidence.dense_score is not None
                    else None
                ),
                "dense_rank": evidence.dense_rank,
                "chunk_order": evidence.chunk_order,
                "heading_path": list(
                    evidence.heading_path
                ),
                "text": evidence.text,
            }
            for evidence in result.evidence
        ],
        "latency_ms": {
            name: round(value, 3)
            for name, value in result.latency_ms.items()
        },
        "warnings": list(result.warnings),
    }

    print(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()