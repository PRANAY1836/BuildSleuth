"""Stage S5 security, configuration, and deployment readiness verification tests."""

from __future__ import annotations

from pathlib import Path


def test_gitignore_protects_secrets_and_sqlite_dbs() -> None:
    gitignore_path = Path(".gitignore")
    assert gitignore_path.exists(), ".gitignore must exist"
    content = gitignore_path.read_text(encoding="utf-8")

    assert ".env" in content
    assert ".streamlit/secrets.toml" in content
    assert "data/*.db" in content or "*.db" in content


def test_secrets_example_contains_no_real_keys() -> None:
    example_path = Path(".streamlit/secrets.toml.example")
    assert example_path.exists(), ".streamlit/secrets.toml.example must exist"
    content = example_path.read_text(encoding="utf-8")

    assert "GEMINI_API_KEY" in content
    assert "AIza" not in content
    assert "ghp_" not in content


def test_requirements_includes_streamlit_and_core_packages() -> None:
    req_path = Path("requirements.txt")
    assert req_path.exists()
    content = req_path.read_text(encoding="utf-8").lower()

    assert "streamlit" in content
    assert "fastapi" in content
    assert "pydantic" in content