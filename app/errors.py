"""Standardized API exception classes and FastAPI exception handlers."""

from __future__ import annotations

import logging
from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class BuildSleuthError(Exception):
    """Base application exception with HTTP status code and machine-readable code."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}


class PayloadTooLargeError(BuildSleuthError):
    """Raised when uploaded file or pasted log exceeds MAX_UPLOAD_BYTES."""

    def __init__(self, max_bytes: int, actual_bytes: int | None = None) -> None:
        super().__init__(
            message=f"Input exceeds maximum allowed size of {max_bytes} bytes.",
            code="PAYLOAD_TOO_LARGE",
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            details={"max_upload_bytes": max_bytes, "actual_bytes": actual_bytes},
        )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach structured JSON exception handlers to the FastAPI instance."""

    @app.exception_handler(BuildSleuthError)
    async def handle_buildsleuth_error(request: Request, exc: BuildSleuthError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.warning("BuildSleuthError [%s] on %s: %s", exc.code, request.url.path, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content=jsonable_encoder(
                {
                    "error": {
                        "code": exc.code,
                        "message": exc.message,
                        "details": exc.details,
                        "request_id": request_id,
                    }
                }
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        safe_errors = jsonable_encoder(exc.errors())
        return JSONResponse(
            status_code=422,
            content={
                "detail": safe_errors,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed.",
                    "details": {"errors": safe_errors},
                    "request_id": request_id,
                },
            },
        )