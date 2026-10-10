"""API routes for ingestion, chunking, indexing, search, and answers."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from evidenceflow.api.dependencies import (
    ApplicationServices,
    get_application_services,
)
from evidenceflow.api.schemas import (
    AnswerRequest,
    AnswerResponse,
    ChunkRequest,
    ChunkResponse,
    ChunkSummary,
    CitationResponse,
    CitationValidationResponse,
    ErrorResponse,
    EvidenceResponse,
    IndexRequest,
    IndexResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    SearchResultResponse,
)
from evidenceflow.ingestion.errors import IngestionError

router = APIRouter(prefix="/api/v1", tags=["pipeline"])

ServicesDependency = Annotated[
    ApplicationServices,
    Depends(get_application_services),
]
ERROR_RESPONSES = {
    400: {
        "model": ErrorResponse,
        "description": "Invalid request or pipeline state",
    },
    404: {
        "model": ErrorResponse,
        "description": "Requested document was not found",
    },
    500: {
        "model": ErrorResponse,
        "description": "Internal pipeline failure",
    },
}


def bad_request(detail: str) -> HTTPException:
    """Create a standard HTTP 400 exception."""

    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={
            "error": "bad_request",
            "detail": detail,
        },
    )


def not_found(detail: str) -> HTTPException:
    """Create a standard HTTP 404 exception."""

    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "error": "not_found",
            "detail": detail,
        },
    )


@router.post(
    "/documents/ingest",
    response_model=IngestResponse,
    responses=ERROR_RESPONSES,
)
def ingest_document(
    request: IngestRequest,
    services: ServicesDependency,
) -> IngestResponse:
    """Validate, extract, fingerprint, and persist one document."""

    try:
        result = services.ingestion_service.ingest(
            Path(request.path)
        )
    except IngestionError as exc:
        raise bad_request(str(exc)) from exc
    except OSError as exc:
        raise bad_request(str(exc)) from exc

    return IngestResponse(
        document_id=result.document.document_id,
        source_name=result.document.source_name,
        source_type=result.document.source_type,
        status=result.document.status.value,
        outcome=result.outcome,
        changed=result.changed,
        page_count=result.page_count,
    )


@router.post(
    "/documents/chunk",
    response_model=ChunkResponse,
    responses=ERROR_RESPONSES,
)
def chunk_document(
    request: ChunkRequest,
    services: ServicesDependency,
) -> ChunkResponse:
    """Generate and persist chunks for an ingested document."""

    if request.overlap >= request.chunk_size:
        raise bad_request(
            "overlap must be smaller than chunk_size"
        )

    chunking_service = services.create_chunking_service(
        chunk_size=request.chunk_size,
        overlap=request.overlap,
    )

    try:
        result = chunking_service.chunk_document(
            request.document_id
        )
    except ValueError as exc:
        detail = str(exc)

        if "not found" in detail.lower():
            raise not_found(detail) from exc

        raise bad_request(detail) from exc

    return ChunkResponse(
        document_id=result.document_id,
        source_name=result.source_name,
        strategy=result.strategy,
        chunk_count=result.chunk_count,
        chunks=[
            ChunkSummary(
                chunk_id=chunk.chunk_id,
                chunk_order=chunk.chunk_order,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                word_count=chunk.word_count,
                token_count=chunk.token_count,
                heading_path=list(chunk.heading_path),
            )
            for chunk in result.chunks
        ],
    )


@router.post(
    "/documents/index",
    response_model=IndexResponse,
    responses=ERROR_RESPONSES,
)
def index_document(
    request: IndexRequest,
    services: ServicesDependency,
) -> IndexResponse:
    """Generate and persist embeddings for a chunked document."""

    try:
        result = services.embedding_index_service.index_document(
            request.document_id
        )
    except ValueError as exc:
        detail = str(exc)

        if "not found" in detail.lower():
            raise not_found(detail) from exc

        raise bad_request(detail) from exc

    return IndexResponse(
        document_id=result.document_id,
        source_name=result.source_name,
        model_name=result.model_name,
        vector_count=result.vector_count,
        dimension=result.dimension,
    )


@router.post(
    "/search/hybrid",
    response_model=SearchResponse,
    responses=ERROR_RESPONSES,
)
def hybrid_search(
    request: SearchRequest,
    services: ServicesDependency,
) -> SearchResponse:
    """Search indexed chunks using BM25, dense retrieval, and RRF."""

    try:
        response = services.hybrid_retriever.search(
            query=request.query,
            candidate_limit=request.candidates,
            result_limit=request.limit,
        )
    except ValueError as exc:
        raise bad_request(str(exc)) from exc

    return SearchResponse(
        query=response.query,
        retrieval_mode="hybrid_rrf",
        result_count=len(response.results),
        latency_ms=response.latency_ms,
        results=[
            SearchResultResponse(
                rank=result.fusion_rank,
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                source_name=result.source_name,
                text=result.text,
                fusion_score=result.fusion_score,
                sparse_score=result.sparse_score,
                sparse_rank=result.sparse_rank,
                dense_score=result.dense_score,
                dense_rank=result.dense_rank,
                heading_path=list(result.heading_path),
            )
            for result in response.results
        ],
    )


@router.post(
    "/answers",
    response_model=AnswerResponse,
    responses=ERROR_RESPONSES,
)
def answer_question(
    request: AnswerRequest,
    services: ServicesDependency,
) -> AnswerResponse:
    """Return a grounded answer or an explicit refusal."""

    if request.limit > request.candidates:
        raise bad_request(
            "limit cannot exceed candidates"
        )

    answer_service = services.create_answer_service(
        minimum_rerank_score=(
            request.minimum_rerank_score
        )
    )

    try:
        result = answer_service.answer(
            question=request.question,
            candidate_limit=request.candidates,
            result_limit=request.limit,
        )
    except ValueError as exc:
        raise bad_request(str(exc)) from exc

    return AnswerResponse(
        question=result.question,
        answer=result.answer,
        supported=result.supported,
        refusal_reason=result.refusal_reason,
        retrieval_mode=result.retrieval_mode,
        reranker_model=result.model_name,
        used_reranker_fallback=(
            result.used_reranker_fallback
        ),
        citation_validation=CitationValidationResponse(
            valid=result.citation_validation.valid,
            coverage=result.citation_validation.coverage,
            valid_citation_ids=list(
                result.citation_validation.valid_citation_ids
            ),
            invalid_citation_ids=list(
                result.citation_validation.invalid_citation_ids
            ),
            missing_citation_ids=list(
                result.citation_validation.missing_citation_ids
            ),
        ),
        citations=[
            CitationResponse(
                citation_id=citation.citation_id,
                chunk_id=citation.chunk_id,
                document_id=citation.document_id,
                source_name=citation.source_name,
                heading_path=list(
                    citation.heading_path
                ),
            )
            for citation in result.citations
        ],
        evidence=[
            EvidenceResponse(
                chunk_id=evidence.chunk_id,
                document_id=evidence.document_id,
                source_name=evidence.source_name,
                rerank_score=evidence.rerank_score,
                rerank_rank=evidence.rerank_rank,
                fusion_rank=evidence.fusion_rank,
                text=evidence.text,
            )
            for evidence in result.evidence
        ],
        latency_ms=result.latency_ms,
        warnings=list(result.warnings),
    )