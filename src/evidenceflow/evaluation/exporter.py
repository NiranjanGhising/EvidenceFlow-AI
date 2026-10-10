"""Export evaluation reports to JSON and CSV files."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from evidenceflow.evaluation.models import EvaluationReport


def export_json(
    report: EvaluationReport,
    output_path: Path,
) -> Path:
    """Export the complete evaluation report as JSON."""

    resolved_path = output_path.expanduser().resolve()

    resolved_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_data = asdict(report)

    resolved_path.write_text(
        json.dumps(
            report_data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return resolved_path


def export_csv(
    report: EvaluationReport,
    output_path: Path,
) -> Path:
    """Export per-question evaluation diagnostics as CSV."""

    resolved_path = output_path.expanduser().resolve()

    resolved_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "run_id",
        "created_at",
        "retrieval_limit",
        "embedding_model",
        "reranker_model",
        "question_id",
        "question",
        "passed",
        "failure_reasons",
        "precision_at_k",
        "recall_at_k",
        "reciprocal_rank",
        "ndcg_at_k",
        "retrieved_chunk_ids",
        "supported",
        "expected_answerable",
        "refusal_correct",
        "unsupported_answer",
        "citation_valid",
        "citation_coverage",
        "returned_citation_ids",
        "retrieval_latency_ms",
        "evidence_check_latency_ms",
        "generation_latency_ms",
        "citation_validation_latency_ms",
        "total_latency_ms",
    ]

    with resolved_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in report.results:
            writer.writerow(
                {
                    "run_id": report.run_id,
                    "created_at": report.created_at,
                    "retrieval_limit": (
                        report.retrieval_limit
                    ),
                    "embedding_model": (
                        report.embedding_model
                    ),
                    "reranker_model": (
                        report.reranker_model
                    ),
                    "question_id": result.question_id,
                    "question": result.question,
                    "passed": result.passed,
                    "failure_reasons": "|".join(
                        result.failure_reasons
                    ),
                    "precision_at_k": (
                        result.retrieval.precision_at_k
                    ),
                    "recall_at_k": (
                        result.retrieval.recall_at_k
                    ),
                    "reciprocal_rank": (
                        result.retrieval.reciprocal_rank
                    ),
                    "ndcg_at_k": (
                        result.retrieval.ndcg_at_k
                    ),
                    "retrieved_chunk_ids": "|".join(
                        result.retrieval.retrieved_chunk_ids
                    ),
                    "supported": (
                        result.answer.supported
                    ),
                    "expected_answerable": (
                        result.answer.expected_answerable
                    ),
                    "refusal_correct": (
                        result.answer.refusal_correct
                    ),
                    "unsupported_answer": (
                        result.answer.unsupported_answer
                    ),
                    "citation_valid": (
                        result.answer.citation_valid
                    ),
                    "citation_coverage": (
                        result.answer.citation_coverage
                    ),
                    "returned_citation_ids": "|".join(
                        result.answer.returned_citation_ids
                    ),
                    "retrieval_latency_ms": (
                        result.latency_ms.get(
                            "retrieval",
                            0.0,
                        )
                    ),
                    "evidence_check_latency_ms": (
                        result.latency_ms.get(
                            "evidence_check",
                            0.0,
                        )
                    ),
                    "generation_latency_ms": (
                        result.latency_ms.get(
                            "generation",
                            0.0,
                        )
                    ),
                    "citation_validation_latency_ms": (
                        result.latency_ms.get(
                            "citation_validation",
                            0.0,
                        )
                    ),
                    "total_latency_ms": (
                        result.latency_ms.get(
                            "total",
                            0.0,
                        )
                    ),
                }
            )

    return resolved_path


def export_summary_json(
    report: EvaluationReport,
    output_path: Path,
) -> Path:
    """Export only aggregate evaluation metrics as JSON."""

    resolved_path = output_path.expanduser().resolve()

    resolved_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_data = {
        "run_id": report.run_id,
        "created_at": report.created_at,
        "retrieval_limit": report.retrieval_limit,
        "embedding_model": report.embedding_model,
        "reranker_model": report.reranker_model,
        "summary": asdict(report.summary),
    }

    resolved_path.write_text(
        json.dumps(
            summary_data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return resolved_path


def export_report_bundle(
    report: EvaluationReport,
    output_directory: Path,
) -> dict[str, Path]:
    """Export complete JSON, summary JSON, and CSV reports."""

    resolved_directory = (
        output_directory.expanduser().resolve()
    )

    resolved_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = export_json(
        report=report,
        output_path=(
            resolved_directory
            / f"{report.run_id}-report.json"
        ),
    )

    summary_path = export_summary_json(
        report=report,
        output_path=(
            resolved_directory
            / f"{report.run_id}-summary.json"
        ),
    )

    csv_path = export_csv(
        report=report,
        output_path=(
            resolved_directory
            / f"{report.run_id}-questions.csv"
        ),
    )

    return {
        "json_report": json_path,
        "summary_report": summary_path,
        "csv_report": csv_path,
    }