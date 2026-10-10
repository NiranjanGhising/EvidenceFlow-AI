import pytest

from evidenceflow.domain.models import Chunk, Document, ProcessingStatus


def test_document_defaults_to_pending() -> None:
    doc = Document("d1", "guide.md", "md", "abc", 100)
    assert doc.status is ProcessingStatus.PENDING


def test_chunk_validates_offsets() -> None:
    with pytest.raises(ValueError, match="offsets"):
        Chunk("c1", "h", "d1", "guide.md", "md", 0, 10, 4, "text", 1, 1, "recursive")


def test_chunk_rejects_blank_text() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        Chunk("c1", "h", "d1", "guide.md", "md", 0, 0, 1, "   ", 0, 0, "recursive")
