"""Command-line evaluation runner."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from evidenceflow.core.config import Settings
from evidenceflow.core.logging import configure_logging
from evidenceflow.evaluation.exporter import export_report_bundle
from evidenceflow.evaluation.loader import load_evaluation_questions
from evidenceflow.evaluation.service import EvaluationRunner
from evidenceflow.generation.cli import build_answer_service


def main() -> None:
    """Run the EvidenceFlow evaluation framework."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate retrieval quality, citation quality, "
            "refusal accuracy, and response latency"
        )
    )

    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to the evaluation dataset JSON file",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/evaluation"),
        help="Directory used for JSON and CSV reports",
    )

    parser.add_argument(
        "--retrieval-limit",
        type=int,
        default=5,
        help="Number of retrieved results evaluated per question",
    )

    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=10,
        help="Number of hybrid candidates sent to the reranker",
    )

    parser.add_argument(
        "--minimum-rerank-score",
        type=float,
        default=0.0,
        help="Minimum reranker score required for answer support",
    )

    arguments = parser.parse_args()

    settings = Settings.from_env()
    settings.ensure_directories()
    configure_logging(settings.log_level)

    questions = load_evaluation_questions(
        arguments.dataset
    )

    answer_service = build_answer_service(
        settings=settings,
        minimum_rerank_score=(
            arguments.minimum_rerank_score
        ),
    )

    runner = EvaluationRunner(
        answer_service=answer_service,
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model,
        retrieval_limit=arguments.retrieval_limit,
        candidate_limit=arguments.candidate_limit,
    )

    report = runner.run(questions)

    exported_files = export_report_bundle(
        report=report,
        output_directory=arguments.output_directory,
    )

    output = {
        "run_id": report.run_id,
        "created_at": report.created_at,
        "question_count": report.summary.question_count,
        "retrieval_limit": report.retrieval_limit,
        "embedding_model": report.embedding_model,
        "reranker_model": report.reranker_model,
        "summary": asdict(report.summary),
        "exported_files": {
            name: str(path)
            for name, path in exported_files.items()
        },
    }

    print(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()