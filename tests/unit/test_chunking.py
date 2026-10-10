import pytest

from evidenceflow.chunking.markdown import split_markdown_sections
from evidenceflow.chunking.recursive import recursive_split
from evidenceflow.chunking.service import (
    chunk_hash,
    chunk_id,
    estimate_token_count,
)


def test_recursive_split_short_text() -> None:
    spans = recursive_split(
        "EvidenceFlow processes technical documentation.",
        chunk_size=100,
        overlap=10,
    )

    assert len(spans) == 1
    assert spans[0].start_char == 0
    assert spans[0].text.startswith("EvidenceFlow")


def test_recursive_split_creates_multiple_chunks() -> None:
    text = " ".join(f"word-{index}" for index in range(100))

    spans = recursive_split(
        text,
        chunk_size=100,
        overlap=20,
    )

    assert len(spans) > 1

    for span in spans:
        assert span.start_char >= 0
        assert span.end_char > span.start_char
        assert span.text


def test_recursive_split_offsets_match_source() -> None:
    text = (
        "First paragraph contains evidence.\n\n"
        "Second paragraph contains more evidence.\n\n"
        "Final paragraph."
    )

    spans = recursive_split(
        text,
        chunk_size=50,
        overlap=10,
    )

    for span in spans:
        assert text[span.start_char : span.end_char] == span.text


def test_invalid_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="smaller"):
        recursive_split(
            "text",
            chunk_size=100,
            overlap=100,
        )


def test_markdown_heading_hierarchy() -> None:
    text = (
        "# Main Guide\n"
        "Introduction.\n\n"
        "## Installation\n"
        "Install the package.\n\n"
        "### Linux\n"
        "Run the Linux command.\n"
    )

    sections = split_markdown_sections(text)

    assert len(sections) == 3
    assert sections[0].heading_path == ("Main Guide",)
    assert sections[1].heading_path == (
        "Main Guide",
        "Installation",
    )
    assert sections[2].heading_path == (
        "Main Guide",
        "Installation",
        "Linux",
    )


def test_markdown_without_headings_returns_whole_document() -> None:
    sections = split_markdown_sections(
        "A plain Markdown document without headings."
    )

    assert len(sections) == 1
    assert sections[0].heading_path == ()


def test_chunk_identifier_is_deterministic() -> None:
    first = chunk_id(
        document_id="doc-1",
        strategy="recursive",
        start_char=0,
        end_char=10,
        text="Evidence",
    )

    second = chunk_id(
        document_id="doc-1",
        strategy="recursive",
        start_char=0,
        end_char=10,
        text="Evidence",
    )

    assert first == second


def test_chunk_hash_changes_with_text() -> None:
    assert chunk_hash("Version one") != chunk_hash("Version two")


def test_token_count_is_positive() -> None:
    assert estimate_token_count(
        "FastAPI validates requests."
    ) >= 4