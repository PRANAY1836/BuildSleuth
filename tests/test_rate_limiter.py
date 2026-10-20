"""Focused unit tests for app.security exports and SQLiteRateLimiter."""

from __future__ import annotations

from pathlib import Path

from app.security import (
    SQLiteRateLimiter,
    normalize_log_text,
    redact_secrets,
    strip_ansi_codes,
)
from app.security.rate_limiter import SQLiteRateLimiter as SubmoduleRateLimiter


def test_security_package_exports_and_normalization() -> None:
    raw = "2026-10-20T10:00:00Z \x1b[31mError:\x1b[0m token=ghp_1234567890abcdefghijklmnopqrstuv"
    cleaned, timestamps, injection = normalize_log_text(raw)

    assert "\x1b[31m" not in cleaned
    assert "[REDACTED_GITHUB_TOKEN]" in cleaned
    assert timestamps == ["2026-10-20T10:00:00Z"]
    assert injection is False
    assert strip_ansi_codes("\x1b[32mOK\x1b[0m") == "OK"
    assert "[REDACTED_GOOGLE_API_KEY]" in redact_secrets(
        "AIzaSyA123456789012345678901234567890"
    )


def test_sqlite_rate_limiter_enforces_sliding_window(tmp_path: Path) -> None:
    db_file = tmp_path / "test_limits.db"
    limiter = SQLiteRateLimiter(db_path=db_file, max_requests=3, window_seconds=60)
    assert SubmoduleRateLimiter is not None

    allowed_1, rem_1 = limiter.check_rate_limit("client_a")
    allowed_2, rem_2 = limiter.check_rate_limit("client_a")
    allowed_3, rem_3 = limiter.check_rate_limit("client_a")
    allowed_4, rem_4 = limiter.check_rate_limit("client_a")

    assert (allowed_1, rem_1) == (True, 2)
    assert (allowed_2, rem_2) == (True, 1)
    assert (allowed_3, rem_3) == (True, 0)
    assert (allowed_4, rem_4) == (False, 0)

    # Separate client has independent quota
    allowed_b, rem_b = limiter.check_rate_limit("client_b")
    assert (allowed_b, rem_b) == (True, 2)