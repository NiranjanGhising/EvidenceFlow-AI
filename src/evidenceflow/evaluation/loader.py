"""Load and validate evaluation questions from JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evidenceflow.evaluation.models import EvaluationQuestion


class EvaluationDatasetError(ValueError):
    """Raised when an evaluation dataset is invalid."""


def _require_non_empty_string(
    record: dict[str, Any],
    field_name: str,
    record_number: int,
) -> str:
    """Return a required non-empty string field."""

    value = record.get(field_name)

    if not isinstance(value, str) or not value.strip():
        raise EvaluationDatasetError(
            f"Record {record_number}: "
            f"'{field_name}' must be a non-empty string"
        )

    return value.strip()


def _parse_string_set(
    value: object,
    field_name: str,
    record_number: int,
) -> frozenset[str]:
    """Parse a JSON list containing unique non-empty strings."""

    if value is None:
        return frozenset()

    if not isinstance(value, list):
        raise EvaluationDatasetError(
            f"Record {record_number}: "
            f"'{field_name}' must be a list"
        )

    parsed_values: set[str] = set()

    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise EvaluationDatasetError(
                f"Record {record_number}: "
                f"'{field_name}' must contain only "
                "non-empty strings"
            )

        parsed_values.add(item.strip())

    return frozenset(parsed_values)


def _parse_metadata(
    value: object,
    record_number: int,
) -> dict[str, Any]:
    """Parse optional evaluation metadata."""

    if value is None:
        return {}

    if not isinstance(value, dict):
        raise EvaluationDatasetError(
            f"Record {record_number}: "
            "'metadata' must be a JSON object"
        )

    return dict(value)


def parse_evaluation_question(
    record: dict[str, Any],
    record_number: int,
) -> EvaluationQuestion:
    """Convert one validated JSON record into an evaluation question."""

    question_id = _require_non_empty_string(
        record=record,
        field_name="question_id",
        record_number=record_number,
    )

    question = _require_non_empty_string(
        record=record,
        field_name="question",
        record_number=record_number,
    )

    answerable = record.get("answerable")

    if not isinstance(answerable, bool):
        raise EvaluationDatasetError(
            f"Record {record_number}: "
            "'answerable' must be true or false"
        )

    relevant_chunk_ids = _parse_string_set(
        value=record.get("relevant_chunk_ids"),
        field_name="relevant_chunk_ids",
        record_number=record_number,
    )

    expected_citation_ids = _parse_string_set(
        value=record.get("expected_citation_ids"),
        field_name="expected_citation_ids",
        record_number=record_number,
    )

    if answerable and not relevant_chunk_ids:
        raise EvaluationDatasetError(
            f"Record {record_number}: "
            "answerable questions must provide at least "
            "one relevant chunk ID"
        )

    if (
        expected_citation_ids
        and not expected_citation_ids.issubset(
            relevant_chunk_ids
        )
    ):
        raise EvaluationDatasetError(
            f"Record {record_number}: "
            "expected citation IDs must also appear in "
            "relevant chunk IDs"
        )

    return EvaluationQuestion(
        question_id=question_id,
        question=question,
        relevant_chunk_ids=relevant_chunk_ids,
        answerable=answerable,
        expected_citation_ids=expected_citation_ids,
        metadata=_parse_metadata(
            value=record.get("metadata"),
            record_number=record_number,
        ),
    )


def load_evaluation_questions(
    path: Path,
) -> list[EvaluationQuestion]:
    """Load a validated evaluation dataset from a JSON file."""

    resolved_path = path.expanduser().resolve()

    if not resolved_path.exists():
        raise EvaluationDatasetError(
            f"Evaluation dataset does not exist: {path}"
        )

    if not resolved_path.is_file():
        raise EvaluationDatasetError(
            f"Evaluation dataset is not a file: {path}"
        )

    try:
        raw_data = json.loads(
            resolved_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise EvaluationDatasetError(
            f"Evaluation dataset contains invalid JSON: {exc}"
        ) from exc

    if not isinstance(raw_data, list):
        raise EvaluationDatasetError(
            "Evaluation dataset must contain a JSON list"
        )

    if not raw_data:
        raise EvaluationDatasetError(
            "Evaluation dataset cannot be empty"
        )

    questions: list[EvaluationQuestion] = []
    seen_question_ids: set[str] = set()

    for record_number, record in enumerate(
        raw_data,
        start=1,
    ):
        if not isinstance(record, dict):
            raise EvaluationDatasetError(
                f"Record {record_number}: "
                "each evaluation record must be a JSON object"
            )

        question = parse_evaluation_question(
            record=record,
            record_number=record_number,
        )

        if question.question_id in seen_question_ids:
            raise EvaluationDatasetError(
                f"Record {record_number}: duplicate question ID "
                f"'{question.question_id}'"
            )

        seen_question_ids.add(question.question_id)
        questions.append(question)

    return questions