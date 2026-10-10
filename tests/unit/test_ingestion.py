from pathlib import Path

import pytest

from evidenceflow.ingestion.errors import (
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from evidenceflow.ingestion.extractors import extract_text_file
from evidenceflow.ingestion.fingerprints import (
    document_id_for_path,
    sha256_bytes,
    sha256_file,
)
from evidenceflow.ingestion.validation import validate_file


def test_sha256_fingerprint_is_stable(tmp_path: Path) -> None:
    document_path = tmp_path / "document.txt"
    document_path.write_bytes(b"EvidenceFlow")

    assert sha256_file(document_path) == sha256_bytes(b"EvidenceFlow")


def test_document_id_is_case_insensitive(tmp_path: Path) -> None:
    first = document_id_for_path(tmp_path / "Guide.MD")
    second = document_id_for_path(tmp_path / "guide.md")

    assert first == second


def test_validate_text_file(tmp_path: Path) -> None:
    document_path = tmp_path / "guide.txt"
    document_path.write_text("content", encoding="utf-8")

    validated = validate_file(document_path, max_upload_mb=1)

    assert validated.source_type == "text"
    assert validated.source_name == "guide.txt"
    assert validated.size_bytes == 7


def test_validate_markdown_file(tmp_path: Path) -> None:
    document_path = tmp_path / "guide.md"
    document_path.write_text("# Guide", encoding="utf-8")

    validated = validate_file(document_path, max_upload_mb=1)

    assert validated.source_type == "markdown"


def test_unsupported_file_type_is_rejected(tmp_path: Path) -> None:
    document_path = tmp_path / "data.csv"
    document_path.write_text("a,b", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError):
        validate_file(document_path, max_upload_mb=1)


def test_text_extraction_normalizes_newlines(tmp_path: Path) -> None:
    document_path = tmp_path / "guide.md"
    document_path.write_bytes(b"# Guide\r\n\r\nDocument text  \r\n")

    extracted = extract_text_file(document_path)

    assert extracted.text == "# Guide\n\nDocument text"


def test_empty_text_file_is_rejected(tmp_path: Path) -> None:
    document_path = tmp_path / "empty.txt"
    document_path.write_text("  \n", encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        extract_text_file(document_path)