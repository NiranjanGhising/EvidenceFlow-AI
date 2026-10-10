"""Command-line document ingestion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidenceflow.core.config import Settings
from evidenceflow.core.logging import configure_logging
from evidenceflow.ingestion.errors import IngestionError
from evidenceflow.ingestion.service import IngestionService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a document into EvidenceFlow AI"
    )

    parser.add_argument(
        "path",
        type=Path,
        help="Path to a PDF, Markdown, or text document",
    )

    arguments = parser.parse_args()

    settings = Settings.from_env()
    settings.ensure_directories()
    configure_logging(settings.log_level)

    try:
        result = IngestionService(settings).ingest(arguments.path)
    except IngestionError as exc:
        parser.exit(status=1, message=f"Ingestion failed: {exc}\n")

    print(
        json.dumps(
            {
                "document_id": result.document.document_id,
                "source_name": result.document.source_name,
                "source_type": result.document.source_type,
                "status": result.document.status.value,
                "outcome": result.outcome,
                "changed": result.changed,
                "page_count": result.page_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()