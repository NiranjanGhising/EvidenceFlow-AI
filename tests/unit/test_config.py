from pathlib import Path

import pytest

from evidenceflow.core.config import Settings


def test_default_settings_use_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(__import__("os").environ):
        if name.startswith("EVIDENCEFLOW_"):
            monkeypatch.delenv(name, raising=False)
    settings = Settings.from_env(tmp_path)
    assert settings.app_name == "EvidenceFlow AI"
    assert settings.data_dir == (tmp_path / "data").resolve()
    assert settings.reranker_enabled is False


def test_environment_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVIDENCEFLOW_ENV", "test")
    monkeypatch.setenv("EVIDENCEFLOW_MAX_UPLOAD_MB", "10")
    monkeypatch.setenv("EVIDENCEFLOW_RERANKER_ENABLED", "true")
    settings = Settings.from_env(tmp_path)
    assert settings.environment == "test"
    assert settings.max_upload_mb == 10
    assert settings.reranker_enabled is True


def test_invalid_upload_limit_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVIDENCEFLOW_MAX_UPLOAD_MB", "0")
    with pytest.raises(ValueError, match="greater than zero"):
        Settings.from_env(tmp_path)


def test_ensure_directories(tmp_path: Path) -> None:
    settings = Settings.from_env(tmp_path)
    settings.ensure_directories()
    assert settings.documents_dir.is_dir()
    assert settings.evaluation_dir.is_dir()
    assert settings.artifacts_dir.is_dir()
