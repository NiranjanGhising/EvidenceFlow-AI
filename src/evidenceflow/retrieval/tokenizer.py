"""Technical-text tokenization for sparse retrieval."""

from __future__ import annotations

import re

TOKEN_PATTERN = re.compile(
    r"""
    [A-Za-z_][A-Za-z0-9_.:/-]*
    |
    \d+(?:\.\d+)*
    """,
    re.VERBOSE,
)


def tokenize_for_search(text: str) -> list[str]:
    """Return normalized tokens while preserving technical identifiers."""

    return [
        match.group(0).lower()
        for match in TOKEN_PATTERN.finditer(text)
    ]