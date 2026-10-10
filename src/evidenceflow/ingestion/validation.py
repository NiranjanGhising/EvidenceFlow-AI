"""Document validation before text extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evidenceflow.ingestion.errors import (
    FileTooLargeError,
    UnsafePathError,
    UnsupportedFileTypeError,
)

SUPPORTED_TYPES = {
    ".txt": "text",
    ".md": "markdown",
    ".pdf": "pdf",
}


@dataclass(frozen=True, slots=True)
class ValidatedFile:
    """Information collected during file validation."""

    path: Path
    source_name: str
    source_type: str
    size_bytes: int


def validate_file(path: Path, max_upload_mb: int) -> ValidatedFile:
    """Validate the path, extension, and file size."""

    resolved_path = path.expanduser().resolve()

    if not resolved_path.exists():
        raise UnsafePathError(f"File does not exist: {path}")

    if not resolved_path.is_file():
        raise UnsafePathError(f"Path is not a regular file: {path}")

    if resolved_path.is_symlink():
        raise UnsafePathError(f"Symbolic links are not accepted: {path}")

    source_type = SUPPORTED_TYPES.get(resolved_path.suffix.lower())

    if source_type is None:
        allowed = ", ".join(sorted(SUPPORTED_TYPES))
        extension = resolved_path.suffix or "<none>"

        raise UnsupportedFileTypeError(
            f"Unsupported file type '{extension}'. Allowed types: {allowed}"
        )

    size_bytes = resolved_path.stat().st_size
    maximum_bytes = max_upload_mb * 1024 * 1024

    if size_bytes > maximum_bytes:
        raise FileTooLargeError(
            f"File size {size_bytes} bytes exceeds the "
            f"configured {max_upload_mb} MB limit"
        )

    return ValidatedFile(
        path=resolved_path,
        source_name=resolved_path.name,
        source_type=source_type,
        size_bytes=size_bytes,
    )