"""Text extraction for PDF, Markdown, and plain-text documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from evidenceflow.ingestion.errors import EmptyDocumentError, IngestionError


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """Text and metadata extracted from a document."""

    text: str
    page_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_text(text: str) -> str:
    """Normalize line endings and trailing whitespace."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines = [line.rstrip() for line in normalized.splitlines()]

    return "\n".join(cleaned_lines).strip()


def extract_text_file(path: Path) -> ExtractedDocument:
    """Extract UTF-8 text from a Markdown or text file."""

    try:
        raw_text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise IngestionError(
            f"File is not valid UTF-8 text: {path.name}"
        ) from exc

    text = normalize_text(raw_text)

    if not text:
        raise EmptyDocumentError(
            f"No usable text was found in {path.name}"
        )

    return ExtractedDocument(text=text)


def extract_pdf(path: Path) -> ExtractedDocument:
    """Extract text and metadata from a text-based PDF."""

    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise IngestionError(
            f"Unable to read PDF document: {path.name}"
        ) from exc

    text = normalize_text("\n\n".join(pages))

    if not text:
        raise EmptyDocumentError(
            f"No extractable text was found in {path.name}. "
            "Image-only PDFs require OCR, which is not enabled yet."
        )

    pdf_metadata = {
        str(key).lstrip("/"): str(value)
        for key, value in (reader.metadata or {}).items()
    }

    return ExtractedDocument(
        text=text,
        page_count=len(reader.pages),
        metadata=pdf_metadata,
    )


def extract_document(path: Path, source_type: str) -> ExtractedDocument:
    """Select the correct extractor for a validated source type."""

    if source_type in {"text", "markdown"}:
        return extract_text_file(path)

    if source_type == "pdf":
        return extract_pdf(path)

    raise IngestionError(
        f"No extractor is registered for source type: {source_type}"
    )