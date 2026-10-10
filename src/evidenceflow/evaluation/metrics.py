"""Retrieval and answer evaluation metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str] | frozenset[str],
    k: int,
) -> float:
    """Calculate retrieval precision at rank K."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    top_k = list(retrieved_ids[:k])

    if not top_k:
        return 0.0

    relevant_count = sum(
        1
        for chunk_id in top_k
        if chunk_id in relevant_ids
    )

    return relevant_count / len(top_k)


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str] | frozenset[str],
    k: int,
) -> float:
    """Calculate retrieval recall at rank K."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_ids:
        return 1.0

    top_k = set(retrieved_ids[:k])

    relevant_count = len(
        top_k.intersection(relevant_ids)
    )

    return relevant_count / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str] | frozenset[str],
) -> float:
    """Return the reciprocal rank of the first relevant result."""

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def discounted_cumulative_gain(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str] | frozenset[str],
    k: int,
) -> float:
    """Calculate binary-relevance DCG at rank K."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    score = 0.0

    for rank, chunk_id in enumerate(
        retrieved_ids[:k],
        start=1,
    ):
        relevance = (
            1.0
            if chunk_id in relevant_ids
            else 0.0
        )

        score += relevance / math.log2(rank + 1)

    return score


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str] | frozenset[str],
    k: int,
) -> float:
    """Calculate normalized discounted cumulative gain at K."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_ids:
        return 1.0

    actual_dcg = discounted_cumulative_gain(
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        k=k,
    )

    ideal_relevant_count = min(
        len(relevant_ids),
        k,
    )

    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(
            1,
            ideal_relevant_count + 1,
        )
    )

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def citation_coverage(
    returned_citation_ids: Sequence[str],
    expected_citation_ids: set[str] | frozenset[str],
) -> float:
    """Calculate expected citation coverage."""

    if not expected_citation_ids:
        return (
            1.0
            if not returned_citation_ids
            else 0.0
        )

    returned_ids = set(returned_citation_ids)

    covered_ids = returned_ids.intersection(
        expected_citation_ids
    )

    return len(covered_ids) / len(
        expected_citation_ids
    )


def refusal_is_correct(
    supported: bool,
    expected_answerable: bool,
) -> bool:
    """Check whether answering or refusing was correct."""

    if expected_answerable:
        return supported

    return not supported


def is_unsupported_answer(
    supported: bool,
    expected_answerable: bool,
) -> bool:
    """Detect an answer returned for an unanswerable question."""

    return supported and not expected_answerable


def safe_mean(values: Sequence[float]) -> float:
    """Calculate a mean while supporting empty input."""

    if not values:
        return 0.0

    return sum(values) / len(values)