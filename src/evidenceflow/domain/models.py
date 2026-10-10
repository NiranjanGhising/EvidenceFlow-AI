"""Core domain records shared across pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class ProcessingStatus(StrEnum):
    PENDING = "pending"
    PROCESSED = "processed"
    FAILED = "failed"
    SKIPPED_DUPLICATE = "skipped_duplicate"


@dataclass(frozen=True, slots=True)
class Document:
    document_id: str
    source_name: str
    source_type: str
    content_hash: str
    size_bytes: int
    status: ProcessingStatus = ProcessingStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    chunk_hash: str
    document_id: str
    source_name: str
    source_type: str
    chunk_order: int
    start_char: int
    end_char: int
    text: str
    word_count: int
    token_count: int
    chunk_strategy: str
    page_number: int | None = None
    heading_path: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.chunk_order < 0:
            raise ValueError("chunk_order cannot be negative")
        if self.start_char < 0 or self.end_char < self.start_char:
            raise ValueError("invalid chunk character offsets")
        if not self.text.strip():
            raise ValueError("chunk text cannot be empty")
