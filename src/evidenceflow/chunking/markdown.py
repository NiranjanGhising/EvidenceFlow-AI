"""Markdown heading detection and structural section extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass

HEADING_PATTERN = re.compile(
    r"^(?P<level>#{1,6})[ \t]+(?P<title>.+?)[ \t]*$",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class MarkdownSection:
    """A Markdown section with hierarchy and source offsets."""

    text: str
    start_char: int
    end_char: int
    heading_path: tuple[str, ...]


def split_markdown_sections(text: str) -> list[MarkdownSection]:
    """Split Markdown content at headings while retaining heading hierarchy."""

    matches = list(HEADING_PATTERN.finditer(text))

    if not matches:
        cleaned = text.strip()

        if not cleaned:
            return []

        start = text.find(cleaned)

        return [
            MarkdownSection(
                text=cleaned,
                start_char=start,
                end_char=start + len(cleaned),
                heading_path=(),
            )
        ]

    sections: list[MarkdownSection] = []
    heading_stack: list[str] = []

    first_heading_start = matches[0].start()
    preamble = text[:first_heading_start].strip()

    if preamble:
        preamble_start = text.find(preamble)

        sections.append(
            MarkdownSection(
                text=preamble,
                start_char=preamble_start,
                end_char=preamble_start + len(preamble),
                heading_path=(),
            )
        )

    for index, match in enumerate(matches):
        level = len(match.group("level"))
        title = match.group("title").strip()

        heading_stack = heading_stack[: level - 1]
        heading_stack.append(title)

        section_start = match.start()

        if index + 1 < len(matches):
            section_end = matches[index + 1].start()
        else:
            section_end = len(text)

        section_text = text[section_start:section_end].strip()

        if not section_text:
            continue

        leading_whitespace = len(
            text[section_start:section_end]
        ) - len(text[section_start:section_end].lstrip())

        actual_start = section_start + leading_whitespace

        sections.append(
            MarkdownSection(
                text=section_text,
                start_char=actual_start,
                end_char=actual_start + len(section_text),
                heading_path=tuple(heading_stack),
            )
        )

    return sections