"""Structured logging configuration with secret scrubbing."""

from __future__ import annotations

import logging
import sys

from app.security import redact_secrets


class SecretScrubbingFilter(logging.Filter):
    """Redacts known credential patterns from log messages before emission."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        scrubbed = redact_secrets(msg)
        if scrubbed != msg:
            record.msg = scrubbed
            record.args = ()
        return True


def configure_logging(level: str = "INFO") -> None:
    """Configure root and application loggers with secret redaction."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
    )
    handler.addFilter(SecretScrubbingFilter())

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)