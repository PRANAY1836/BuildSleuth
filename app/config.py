"""Centralized, validated configuration management for BuildSleuth."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    APP_ENV: Literal["development", "test", "production"] = "development"
    APP_NAME: str = "BuildSleuth API"
    APP_VERSION: str = "0.2.0"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    ALLOWED_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000"
    MAX_UPLOAD_BYTES: int = Field(default=5 * 1024 * 1024, ge=512, le=50 * 1024 * 1024)
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(default=60, ge=1, le=10000)

    DATA_DIR: Path = Path("./data")
    SQLITE_DB_PATH: Path = Path("./data/buildsleuth.db")

    ENABLE_DENSE_RETRIEVAL: bool = True
    ENABLE_RERANKER: bool = True
    ENABLE_GRAPH_RAG: bool = True
    RRF_K: int = Field(default=60, ge=1, le=200)
    MAX_AGENT_RETRIES: int = Field(default=2, ge=0, le=5)

    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GITHUB_TOKEN: str = ""
    NEO4J_URI: str = ""
    NEO4J_USERNAME: str = ""
    NEO4J_PASSWORD: str = ""

    @field_validator("ALLOWED_ORIGINS")
    @classmethod
    def validate_origins(cls, value: str) -> str:
        origins = [origin.strip() for origin in value.split(",") if origin.strip()]
        if not origins:
            raise ValueError("ALLOWED_ORIGINS must specify at least one valid origin.")
        return ",".join(origins)

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

    @property
    def gemini_configured(self) -> bool:
        return bool(self.GEMINI_API_KEY and self.GEMINI_API_KEY.strip())

    @property
    def github_configured(self) -> bool:
        return bool(self.GITHUB_TOKEN and self.GITHUB_TOKEN.strip())

    @property
    def neo4j_configured(self) -> bool:
        return bool(self.NEO4J_URI and self.NEO4J_URI.strip())

    def safe_diagnostic_summary(self) -> dict[str, object]:
        """Return configuration status safe for UI display without leaking secrets."""
        return {
            "app_env": self.APP_ENV,
            "app_version": self.APP_VERSION,
            "max_upload_bytes": self.MAX_UPLOAD_BYTES,
            "rate_limit_rpm": self.RATE_LIMIT_REQUESTS_PER_MINUTE,
            "features": {
                "dense_retrieval_enabled": self.ENABLE_DENSE_RETRIEVAL,
                "reranker_enabled": self.ENABLE_RERANKER,
                "graph_rag_enabled": self.ENABLE_GRAPH_RAG,
                "rrf_k": self.RRF_K,
                "max_agent_retries": self.MAX_AGENT_RETRIES,
            },
            "integrations": {
                "gemini_api_configured": self.gemini_configured,
                "gemini_model": self.GEMINI_MODEL,
                "github_token_configured": self.github_configured,
                "graph_backend": "neo4j" if self.neo4j_configured else "networkx_sqlite_local",
            },
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()