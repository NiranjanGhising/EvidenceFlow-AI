"""Request and response schemas for the EvidenceFlow API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str
    service: str
    version: str
    environment: str


class IngestRequest(BaseModel):
    """Request to ingest a document from a local path."""

    path: str = Field(
        min_length=1,
        description="Path to a PDF, Markdown, or text file",
    )


class IngestResponse(BaseModel):
    """Document ingestion response."""

    document_id: str
    source_name: str
    source_type: str
    status: str
    outcome: str
    changed: bool
    page_count: int | None


class ChunkRequest(BaseModel):
    """Request to chunk an ingested document."""

    document_id: str = Field(min_length=1)
    chunk_size: int = Field(
        default=800,
        greater_than=0,
    )
    overlap: int = Field(
        default=120,
        greater_than_or_equal=0,
    )


class ChunkSummary(BaseModel):
    """Summary of one generated chunk."""

    chunk_id: str
    chunk_order: int
    start_char: int
    end_char: int
    word_count: int
    token_count: int
    heading_path: list[str]


class ChunkResponse(BaseModel):
    """Document chunking response."""

    document_id: str
    source_name: str
    strategy: str
    chunk_count: int
    chunks: list[ChunkSummary]


class IndexRequest(BaseModel):
    """Request to index a chunked document."""

    document_id: str = Field(min_length=1)


class IndexResponse(BaseModel):
    """Embedding indexing response."""

    document_id: str
    source_name: str
    model_name: str
    vector_count: int
    dimension: int


class SearchRequest(BaseModel):
    """Request for document-scoped hybrid retrieval."""

    query: str = Field(min_length=1)

    document_id: str | None = Field(
        default=None,
        description=(
            "Optional document ID used to restrict retrieval "
            "to one document"
        ),
    )

    limit: int = Field(
        default=5,
        greater_than=0,
    )

    candidates: int = Field(
        default=20,
        greater_than=0,
    )

class SearchResultResponse(BaseModel):
    """One retrieval result."""

    rank: int
    chunk_id: str
    document_id: str
    source_name: str
    text: str
    score: float | None = None
    fusion_score: float | None = None
    sparse_score: float | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    dense_rank: int | None = None
    heading_path: list[str]


class SearchResponse(BaseModel):
    """Search results returned by the API."""

    query: str
    retrieval_mode: str
    result_count: int
    latency_ms: dict[str, float]
    results: list[SearchResultResponse]


class AnswerRequest(BaseModel):
    """Request for a document-scoped grounded answer."""

    question: str = Field(min_length=1)

    document_id: str | None = Field(
        default=None,
        description=(
            "Optional document ID used to restrict retrieval "
            "and answering to one document"
        ),
    )

    conversation_history: list[dict[str, str]] = Field(
        default_factory=list,
        description=(
            "Previous user and assistant messages used "
            "for conversational context"
        ),
    )

    candidates: int = Field(
        default=10,
        greater_than=0,
    )

    limit: int = Field(
        default=5,
        greater_than=0,
    )

    minimum_rerank_score: float | None = Field(
        default=None,
        description=(
            "Optional advanced reranker threshold. "
            "When omitted, retrieval signals determine "
            "evidence sufficiency."
        ),
    )
class CitationResponse(BaseModel):
    """Structured answer citation."""

    citation_id: str
    chunk_id: str
    document_id: str
    source_name: str
    heading_path: list[str]


class CitationValidationResponse(BaseModel):
    """Citation-validation summary."""

    valid: bool
    coverage: float
    valid_citation_ids: list[str]
    invalid_citation_ids: list[str]
    missing_citation_ids: list[str]


class EvidenceResponse(BaseModel):
    """Evidence used to construct an answer."""

    chunk_id: str
    document_id: str
    source_name: str
    rerank_score: float
    rerank_rank: int
    fusion_rank: int
    text: str


class AnswerResponse(BaseModel):
    """Grounded LLM answer API response."""

    question: str
    answer: str
    supported: bool
    refusal_reason: str | None
    retrieval_mode: str
    reranker_model: str
    used_reranker_fallback: bool
    generation_provider: str
    generation_model: str
    used_llm_fallback: bool
    llm_fallback_reason: str | None
    citation_validation: CitationValidationResponse
    citations: list[CitationResponse]
    evidence: list[EvidenceResponse]
    latency_ms: dict[str, float]
    warnings: list[str]

class ErrorResponse(BaseModel):
    """Standard API error response."""

    error: str
    detail: str