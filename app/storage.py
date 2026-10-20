"""SQLite persistence layer for investigations and idempotent evidence storage."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.models import DiagnoseResponse


class InvestigationStore:
    """Lightweight SQLite store for saving and retrieving CI failure investigations."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS investigations (
                    investigation_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    repository TEXT,
                    run_id TEXT,
                    commit_sha TEXT,
                    failure_category TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    confidence_level TEXT NOT NULL,
                    response_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save_investigation(self, response: DiagnoseResponse) -> DiagnoseResponse:
        inv_id = response.investigation_id or f"inv_{uuid.uuid4().hex[:12]}"
        updated = response.model_copy(update={"investigation_id": inv_id})
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO investigations (
                    investigation_id, created_at, repository, run_id, commit_sha,
                    failure_category, summary, confidence_level, response_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    inv_id,
                    now_iso,
                    updated.repository,
                    updated.run_id,
                    updated.commit_sha,
                    updated.failure_category,
                    updated.summary,
                    updated.confidence_level,
                    updated.model_dump_json(),
                ),
            )
            conn.commit()
        return updated

    def get_investigation(self, investigation_id: str) -> DiagnoseResponse | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT response_json FROM investigations WHERE investigation_id = ?",
                (investigation_id,),
            ).fetchone()
            if row is None:
                return None
            return DiagnoseResponse.model_validate_json(row["response_json"])

    def list_investigations(self, limit: int = 20) -> list[dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT investigation_id, created_at, repository, run_id,
                       failure_category, summary, confidence_level
                FROM investigations
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]