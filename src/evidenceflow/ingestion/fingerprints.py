"""Document fingerprint and identifier utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_bytes(content: bytes) -> str:
    """Return the SHA-256 digest of byte content."""

    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Calculate a SHA-256 digest without loading the full file into memory."""

    digest = hashlib.sha256()

    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)

    return digest.hexdigest()


def document_id_for_path(path: Path) -> str:
    """Create a stable document ID from the normalized filename."""

    normalized_name = path.name.strip().lower().encode("utf-8")
    digest = hashlib.sha256(normalized_name).hexdigest()[:16]

    return f"doc-{digest}"