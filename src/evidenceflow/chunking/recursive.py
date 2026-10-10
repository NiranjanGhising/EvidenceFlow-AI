"""Recursive character-aware text splitting."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SEPARATORS = (
    "\n\n",
    "\n",
    ". ",
    "! ",
    "? ",
    "; ",
    ", ",
    " ",
)


@dataclass(frozen=True, slots=True)
class TextSpan:
    """Text content with source character offsets."""

    text: str
    start_char: int
    end_char: int


def _find_split_position(
    text: str,
    start: int,
    maximum_end: int,
    separators: tuple[str, ...],
) -> int:
    """Find the best separator before the chunk-size boundary."""

    window = text[start:maximum_end]

    for separator in separators:
        position = window.rfind(separator)

        if position <= 0:
            continue

        return start + position + len(separator)

    return maximum_end


def _next_start(
    text: str,
    current_start: int,
    current_end: int,
    overlap: int,
) -> int:
    """Calculate the next chunk start while preserving overlap."""

    if overlap <= 0:
        return current_end

    proposed_start = max(current_start + 1, current_end - overlap)

    while (
        proposed_start < current_end
        and proposed_start > 0
        and not text[proposed_start - 1].isspace()
        and not text[proposed_start].isspace()
    ):
        proposed_start += 1

    while proposed_start < len(text) and text[proposed_start].isspace():
        proposed_start += 1

    return min(proposed_start, current_end)


def recursive_split(
    text: str,
    chunk_size: int = 800,
    overlap: int = 120,
    base_offset: int = 0,
    separators: tuple[str, ...] = DEFAULT_SEPARATORS,
) -> list[TextSpan]:
    """Split text into overlapping chunks with stable character offsets."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if base_offset < 0:
        raise ValueError("base_offset cannot be negative")

    if not text.strip():
        return []

    spans: list[TextSpan] = []
    cursor = 0
    text_length = len(text)

    while cursor < text_length:
        while cursor < text_length and text[cursor].isspace():
            cursor += 1

        if cursor >= text_length:
            break

        maximum_end = min(cursor + chunk_size, text_length)

        if maximum_end < text_length:
            split_end = _find_split_position(
                text=text,
                start=cursor,
                maximum_end=maximum_end,
                separators=separators,
            )
        else:
            split_end = text_length

        if split_end <= cursor:
            split_end = maximum_end

        raw_chunk = text[cursor:split_end]
        chunk_text = raw_chunk.strip()

        if chunk_text:
            leading_whitespace = len(raw_chunk) - len(raw_chunk.lstrip())
            actual_start = cursor + leading_whitespace
            actual_end = actual_start + len(chunk_text)

            spans.append(
                TextSpan(
                    text=chunk_text,
                    start_char=base_offset + actual_start,
                    end_char=base_offset + actual_end,
                )
            )

        if split_end >= text_length:
            break

        next_cursor = _next_start(
            text=text,
            current_start=cursor,
            current_end=split_end,
            overlap=overlap,
        )

        if next_cursor <= cursor:
            next_cursor = split_end

        cursor = next_cursor

    return spans