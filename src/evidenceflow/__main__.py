"""Phase 1 configuration and health smoke check."""

from __future__ import annotations

import json

from evidenceflow import __version__
from evidenceflow.core.config import Settings
from evidenceflow.core.logging import configure_logging


def main() -> None:
    settings = Settings.from_env()
    settings.ensure_directories()
    configure_logging(settings.log_level)
    print(json.dumps({
        "status": "ok",
        "app": settings.app_name,
        "version": __version__,
        "environment": settings.environment,
        "project_root": str(settings.project_root),
    }, indent=2))


if __name__ == "__main__":
    main()
