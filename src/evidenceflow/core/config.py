"""Typed application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

TRUTHY = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in TRUTHY


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str
    environment: str
    debug: bool
    log_level: str
    project_root: Path
    data_dir: Path
    documents_dir: Path
    evaluation_dir: Path
    artifacts_dir: Path
    database_path: Path
    max_upload_mb: int
    embedding_model: str
    reranker_model: str
    reranker_enabled: bool

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> Settings:
        root = (project_root or Path(__file__).resolve().parents[3]).resolve()
        data_dir = Path(os.getenv("EVIDENCEFLOW_DATA_DIR", root / "data")).resolve()
        artifacts_dir = Path(os.getenv("EVIDENCEFLOW_ARTIFACTS_DIR", root / "artifacts")).resolve()
        database_path = Path(
            os.getenv("EVIDENCEFLOW_DATABASE_PATH", artifacts_dir / "evidenceflow.db")
        ).resolve()

        settings = cls(
            app_name=os.getenv("EVIDENCEFLOW_APP_NAME", "EvidenceFlow AI"),
            environment=os.getenv("EVIDENCEFLOW_ENV", "development"),
            debug=_env_bool("EVIDENCEFLOW_DEBUG", False),
            log_level=os.getenv("EVIDENCEFLOW_LOG_LEVEL", "INFO").upper(),
            project_root=root,
            data_dir=data_dir,
            documents_dir=data_dir / "documents",
            evaluation_dir=data_dir / "evaluation",
            artifacts_dir=artifacts_dir,
            database_path=database_path,
            max_upload_mb=_env_int("EVIDENCEFLOW_MAX_UPLOAD_MB", 25),
            embedding_model=os.getenv(
                "EVIDENCEFLOW_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            reranker_model=os.getenv(
                "EVIDENCEFLOW_RERANKER_MODEL",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
            ),
            reranker_enabled=_env_bool("EVIDENCEFLOW_RERANKER_ENABLED", False),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.max_upload_mb <= 0:
            raise ValueError("EVIDENCEFLOW_MAX_UPLOAD_MB must be greater than zero")
        if not self.app_name.strip():
            raise ValueError("EVIDENCEFLOW_APP_NAME cannot be empty")
        if not self.embedding_model.strip():
            raise ValueError("EVIDENCEFLOW_EMBEDDING_MODEL cannot be empty")

    def ensure_directories(self) -> None:
        for directory in (
            self.data_dir,
            self.documents_dir,
            self.evaluation_dir,
            self.artifacts_dir,
            self.database_path.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)
