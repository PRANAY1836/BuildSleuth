"""Stage 7: GitHub client SSRF protection and commit diff endpoint tests."""

from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from app.config import Settings
from app.github.client import GitHubClient
from app.main import create_app


def get_client(tmp_path: Path) -> TestClient:
    settings = Settings(
        APP_ENV="test",
        DATA_DIR=tmp_path,
        SQLITE_DB_PATH=tmp_path / "test_stage7.db",
    )
    return TestClient(create_app(settings_override=settings))


def test_github_client_ssrf_protection() -> None:
    client = GitHubClient()
    
    # Valid formats
    assert client.validate_repository_url("PRANAY1836/BuildSleuth") == "PRANAY1836/BuildSleuth"
    assert client.validate_repository_url("https://github.com/PRANAY1836/BuildSleuth") == "PRANAY1836/BuildSleuth"

    # Invalid / malicious hosts should raise ValueError
    import pytest
    with pytest.raises(ValueError):
        client.validate_repository_url("https://malicious-server.com/foo/bar")


def test_github_diff_endpoint_validation(tmp_path: Path) -> None:
    client = get_client(tmp_path)
    resp = client.get("/api/v1/github/diff?repo=invalid-host.com/bad/repo&commit=12345")
    assert resp.status_code == 400