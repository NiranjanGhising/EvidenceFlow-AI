"""Shared application services and model lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from evidenceflow.chunking.service import ChunkingService
from evidenceflow.core.config import Settings
from evidenceflow.embeddings.encoder import SentenceTransformerEncoder
from evidenceflow.generation.service import (
    EvidenceSufficiencyChecker,
    ExtractiveAnswerGenerator,
    GroundedAnswerService,
)
from evidenceflow.indexing.service import EmbeddingIndexService
from evidenceflow.ingestion.service import IngestionService
from evidenceflow.llm.factory import build_llm_answer_generator
from evidenceflow.llm.generator import LLMAnswerGenerator
from evidenceflow.reranking.encoder import CrossEncoderReranker
from evidenceflow.reranking.service import (
    RerankedRetriever,
    RerankingService,
)
from evidenceflow.retrieval.service import HybridRetriever
from evidenceflow.retrieval.sparse import BM25Retriever
from evidenceflow.storage.chunks import ChunkRepository
from evidenceflow.storage.documents import DocumentRepository
from evidenceflow.storage.vectors import VectorRepository


@dataclass(slots=True)
class ApplicationServices:
    """Long-lived services shared by API requests."""

    settings: Settings
    document_repository: DocumentRepository
    chunk_repository: ChunkRepository
    vector_repository: VectorRepository
    ingestion_service: IngestionService
    embedding_index_service: EmbeddingIndexService
    hybrid_retriever: HybridRetriever
    reranking_service: RerankingService
    reranked_retriever: RerankedRetriever
    llm_answer_generator: LLMAnswerGenerator

    def create_chunking_service(
        self,
        chunk_size: int,
        overlap: int,
    ) -> ChunkingService:
        """Create a chunking service with request-specific settings."""

        return ChunkingService(
            document_repository=self.document_repository,
            chunk_repository=self.chunk_repository,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    def create_answer_service(
        self,
        minimum_rerank_score: float,
    ) -> GroundedAnswerService:
        """Create an LLM answer service with a custom threshold."""

        return GroundedAnswerService(
            retriever=self.reranked_retriever,
            sufficiency_checker=EvidenceSufficiencyChecker(
                minimum_rerank_score=minimum_rerank_score,
                maximum_evidence=2,
            ),
            answer_generator=self.llm_answer_generator,
            fallback_generator=ExtractiveAnswerGenerator(),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    settings = Settings.from_env()
    settings.ensure_directories()

    return settings


@lru_cache(maxsize=1)
def get_application_services() -> ApplicationServices:
    """Create and cache repositories, models, and services."""

    settings = get_settings()

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
    vector_repository.initialize()

    ingestion_service = IngestionService(
        settings=settings,
        repository=document_repository,
    )

    embedding_encoder = SentenceTransformerEncoder(
        model_name=settings.embedding_model,
        device="cpu",
    )

    embedding_index_service = EmbeddingIndexService(
        document_repository=document_repository,
        chunk_repository=chunk_repository,
        vector_repository=vector_repository,
        encoder=embedding_encoder,
    )

    sparse_retriever = BM25Retriever(
        chunk_repository=chunk_repository,
    )

    hybrid_retriever = HybridRetriever(
        sparse_retriever=sparse_retriever,
        dense_retriever=embedding_index_service,
    )

    cross_encoder = CrossEncoderReranker(
        model_name=settings.reranker_model,
        device="cpu",
    )

    reranking_service = RerankingService(
        reranker=cross_encoder,
        fallback_enabled=True,
    )

    reranked_retriever = RerankedRetriever(
        hybrid_retriever=hybrid_retriever,
        reranking_service=reranking_service,
    )

    llm_answer_generator = build_llm_answer_generator()

    return ApplicationServices(
        settings=settings,
        document_repository=document_repository,
        chunk_repository=chunk_repository,
        vector_repository=vector_repository,
        ingestion_service=ingestion_service,
        embedding_index_service=embedding_index_service,
        hybrid_retriever=hybrid_retriever,
        reranking_service=reranking_service,
        reranked_retriever=reranked_retriever,
        llm_answer_generator=llm_answer_generator,
    )


def clear_application_caches() -> None:
    """Clear cached dependencies after configuration changes."""

    get_application_services.cache_clear()
    get_settings.cache_clear()