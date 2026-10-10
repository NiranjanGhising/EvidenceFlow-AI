"""Structure-aware chunk creation and persistence."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

from evidenceflow.chunking.markdown import split_markdown_sections
from evidenceflow.chunking.recursive import TextSpan, recursive_split
from evidenceflow.domain.models import Chunk, Document
from evidenceflow.storage.chunks import ChunkRepository
from evidenceflow.storage.documents import DocumentRepository

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True, slots=True)
class ChunkingResult:
    """Summary returned after chunking one document."""

    document_id: str
    source_name: str
    chunk_count: int
    strategy: str
    chunks: tuple[Chunk, ...]


def estimate_token_count(text: str) -> int:
    """Return a deterministic tokenizer-independent approximation."""

    return len(TOKEN_PATTERN.findall(text))


def chunk_hash(text: str) -> str:
    """Return a stable SHA-256 hash for chunk content."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_id(
    document_id: str,
    strategy: str,
    start_char: int,
    end_char: int,
    text: str,
) -> str:
    """Create a deterministic chunk identifier."""

    identity = (
        f"{document_id}|{strategy}|{start_char}|{end_char}|{text}"
    ).encode()

    digest = hashlib.sha256(identity).hexdigest()[:20]

    return f"chunk-{digest}"


class ChunkingService:
    """Transform extracted document text into persistent chunks."""

    def __init__(
        self,
        document_repository: DocumentRepository,
        chunk_repository: ChunkRepository,
        chunk_size: int = 800,
        overlap: int = 120,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                "overlap must be non-negative and smaller than chunk_size"
            )

        self.document_repository = document_repository
        self.chunk_repository = chunk_repository
        self.chunk_size = chunk_size
        self.overlap = overlap

        self.chunk_repository.initialize()

    def chunk_document(self, document_id: str) -> ChunkingResult:
        """Chunk one previously ingested document."""

        document = self.document_repository.find_by_id(document_id)

        if document is None:
            raise ValueError(f"Document was not found: {document_id}")

        text = self.document_repository.get_text(document_id)

        if text is None or not text.strip():
            raise ValueError(
                f"Document contains no persisted text: {document_id}"
            )

        chunks = self._create_chunks(document, text)

        self.chunk_repository.replace_for_document(
            document_id=document.document_id,
            chunks=chunks,
        )

        logger.info(
            "document_chunked",
            extra={
                "document_id": document.document_id,
                "source_name": document.source_name,
                "chunk_count": len(chunks),
                "strategy": "structure-aware-recursive",
            },
        )

        return ChunkingResult(
            document_id=document.document_id,
            source_name=document.source_name,
            chunk_count=len(chunks),
            strategy="structure-aware-recursive",
            chunks=tuple(chunks),
        )

    def _create_chunks(
        self,
        document: Document,
        text: str,
    ) -> list[Chunk]:
        if document.source_type == "markdown":
            return self._chunk_markdown(document, text)

        spans = recursive_split(
            text=text,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )

        return self._build_chunks(
            document=document,
            spans=spans,
            strategy="recursive",
        )

    def _chunk_markdown(
        self,
        document: Document,
        text: str,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_order = 0

        for section in split_markdown_sections(text):
            spans = recursive_split(
                text=section.text,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
                base_offset=section.start_char,
            )

            section_chunks = self._build_chunks(
                document=document,
                spans=spans,
                strategy="markdown-recursive",
                heading_path=section.heading_path,
                starting_order=chunk_order,
            )

            chunks.extend(section_chunks)
            chunk_order += len(section_chunks)

        return chunks

    @staticmethod
    def _build_chunks(
        document: Document,
        spans: list[TextSpan],
        strategy: str,
        heading_path: tuple[str, ...] = (),
        starting_order: int = 0,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []

        for index, span in enumerate(spans):
            chunks.append(
                Chunk(
                    chunk_id=chunk_id(
                        document_id=document.document_id,
                        strategy=strategy,
                        start_char=span.start_char,
                        end_char=span.end_char,
                        text=span.text,
                    ),
                    chunk_hash=chunk_hash(span.text),
                    document_id=document.document_id,
                    source_name=document.source_name,
                    source_type=document.source_type,
                    chunk_order=starting_order + index,
                    start_char=span.start_char,
                    end_char=span.end_char,
                    text=span.text,
                    word_count=len(span.text.split()),
                    token_count=estimate_token_count(span.text),
                    chunk_strategy=strategy,
                    heading_path=heading_path,
                )
            )

        return chunks