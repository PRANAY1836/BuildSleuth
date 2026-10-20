"""Security, sanitization, and rate limiting package."""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
TIMESTAMP_PREFIX_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)\s+(?P<body>.*)$"
)

SECRET_REDACTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "[REDACTED_GITHUB_PAT]"),
    (re.compile(r"AIza[0-9A-Za-z\-_]{30,}"), "[REDACTED_GOOGLE_API_KEY]"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED_AWS_ACCESS_KEY]"),
    (re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9\-._~+/]+=*"), r"\1[REDACTED_BEARER_TOKEN]"),
    (
        re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
        r"\1=[REDACTED_SECRET]",
    ),
)

PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"system\s+prompt\s*:", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+in\s+developer\s+mode", re.IGNORECASE),
)


def redact_secrets(text: str) -> str:
    """Redact common API keys, tokens, and credentials from untrusted text."""
    sanitized = text
    for pattern, replacement in SECRET_REDACTION_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def strip_ansi_codes(text: str) -> str:
    """Remove terminal color and cursor control escape sequences."""
    return ANSI_ESCAPE_RE.sub("", text)


def normalize_log_text(raw_log: str) -> tuple[str, list[str | None], bool]:
    """Normalize line endings, strip ANSI codes, redact secrets, and extract line timestamps."""
    cleaned = strip_ansi_codes(raw_log.replace("\r\n", "\n").replace("\r", "\n"))
    cleaned = redact_secrets(cleaned)
    injection_detected = any(p.search(cleaned) for p in PROMPT_INJECTION_PATTERNS)

    timestamps: list[str | None] = []
    normalized_lines: list[str] = []
    for line in cleaned.split("\n"):
        m = TIMESTAMP_PREFIX_RE.match(line)
        if m:
            timestamps.append(m.group("ts"))
            normalized_lines.append(m.group("body"))
        else:
            timestamps.append(None)
            normalized_lines.append(line)

    return "\n".join(normalized_lines), timestamps, injection_detected


class SQLiteRateLimiter:
    """Tracks request timestamps per client identifier in SQLite."""

    def __init__(
        self, db_path: Path, max_requests: int = 15, window_seconds: int = 900
    ) -> None:
        self.db_path = db_path
        self.max_requests = max_requests
        self.window_seconds = window_seconds
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
                CREATE TABLE IF NOT EXISTS request_logs (
                    client_id TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
                """
            )
            conn.commit()

    def check_rate_limit(self, client_id: str) -> tuple[bool, int]:
        """Check if client is within rate limits."""
        now = time.time()
        window_start = now - self.window_seconds

        with self._connect() as conn:
            conn.execute(
                "DELETE FROM request_logs WHERE timestamp < ?", (window_start,)
            )

            cursor = conn.execute(
                "SELECT COUNT(*) as count FROM request_logs WHERE client_id = ? AND timestamp >= ?",
                (client_id, window_start),
            )
            row = cursor.fetchone()
            current_count = row["count"] if row else 0

            if current_count >= self.max_requests:
                conn.commit()
                return False, 0

            conn.execute(
                "INSERT INTO request_logs (client_id, timestamp) VALUES (?, ?)",
                (client_id, now),
            )
            conn.commit()

        remaining = max(0, self.max_requests - current_count - 1)
        return True, remaining