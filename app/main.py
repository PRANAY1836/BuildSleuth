"""FastAPI application entrypoint for BuildSleuth."""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent.workflow import run_agentic_investigation
from app.config import Settings, get_settings
from app.diagnosis import diagnose_full
from app.errors import PayloadTooLargeError, register_exception_handlers
from app.github.client import GitHubClient
from app.graph.graph_store import GraphEdge, GraphEntity, GraphStore
from app.hybrid_rag import run_hybrid_retrieval
from app.logging_config import configure_logging
from app.models import (
    DiagnoseRequest,
    DiagnoseResponse,
    HealthResponse,
    ReadinessResponse,
    RetrievalInspectRequest,
    RetrievalInspectResponse,
)
from app.storage import InvestigationStore


def create_app(settings_override: Settings | None = None) -> FastAPI:
    """Create and configure the BuildSleuth FastAPI application."""
    settings = settings_override or get_settings()
    configure_logging(settings.LOG_LEVEL)
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    store = InvestigationStore(settings.SQLITE_DB_PATH)
    graph_store = GraphStore(settings.DATA_DIR / "graph.db")
    github_client = GitHubClient(token=settings.GITHUB_TOKEN)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
        yield

    app = FastAPI(
        title=settings.APP_NAME,
        description="Evidence-grounded CI failure intelligence platform.",
        version=settings.APP_VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
    )

    @app.middleware("http")
    async def security_and_limits_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit():
            actual_bytes = int(content_length)
            if actual_bytes > settings.MAX_UPLOAD_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": {
                            "code": "PAYLOAD_TOO_LARGE",
                            "message": f"Request body ({actual_bytes} bytes) exceeds limit of {settings.MAX_UPLOAD_BYTES} bytes.",
                            "details": {
                                "max_upload_bytes": settings.MAX_UPLOAD_BYTES,
                                "actual_bytes": actual_bytes,
                            },
                            "request_id": request_id,
                        }
                    },
                )

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    register_exception_handlers(app)

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", service="buildsleuth", version=app.version)

    @app.get("/ready", response_model=ReadinessResponse, tags=["system"])
    def readiness_check() -> ReadinessResponse:
        checks = {
            "dense_semantic_retriever": settings.ENABLE_DENSE_RETRIEVAL,
            "sparse_bm25_retriever": True,
            "rrf_fusion": True,
            "reranker": settings.ENABLE_RERANKER,
            "sqlite_store": settings.SQLITE_DB_PATH.exists(),
            "graph_store": (settings.DATA_DIR / "graph.db").exists(),
        }
        return ReadinessResponse(
            status="ready" if all(checks.values()) else "degraded",
            checks=checks,
            details=settings.safe_diagnostic_summary(),
        )

    @app.get("/api/v1/status", tags=["system"])
    def system_status() -> dict[str, object]:
        return settings.safe_diagnostic_summary()

    @app.post("/diagnose", response_model=DiagnoseResponse, tags=["diagnosis"])
    @app.post("/api/v1/diagnose", response_model=DiagnoseResponse, tags=["diagnosis"])
    def diagnose_endpoint(request: DiagnoseRequest) -> DiagnoseResponse:
        raw_log = request.log or request.log_text or ""
        encoded_len = len(raw_log.encode("utf-8"))
        if encoded_len > settings.MAX_UPLOAD_BYTES:
            raise PayloadTooLargeError(
                max_bytes=settings.MAX_UPLOAD_BYTES, actual_bytes=encoded_len
            )
        result = diagnose_full(request, settings=settings)
        return store.save_investigation(result)

    @app.post(
        "/api/v1/retrieval/inspect",
        response_model=RetrievalInspectResponse,
        tags=["retrieval"],
    )
    def inspect_retrieval_endpoint(
        request: RetrievalInspectRequest,
    ) -> RetrievalInspectResponse:
        encoded_len = len(request.log_text.encode("utf-8"))
        if encoded_len > settings.MAX_UPLOAD_BYTES:
            raise PayloadTooLargeError(
                max_bytes=settings.MAX_UPLOAD_BYTES, actual_bytes=encoded_len
            )
        inspection, _inj = run_hybrid_retrieval(request)
        return inspection

    @app.post("/api/v1/agent/investigate", tags=["agent"])
    def agent_investigate_endpoint(request: DiagnoseRequest) -> dict[str, object]:
        raw_log = request.log or request.log_text or ""
        state = run_agentic_investigation(
            log_text=raw_log,
            question=request.question,
            diff=request.diff,
            repository=request.repository,
            commit_sha=request.commit_sha,
            max_retries=settings.MAX_AGENT_RETRIES,
        )
        return {
            "query": state.query,
            "sufficiency": state.sufficiency,
            "reformulation_count": state.reformulation_count,
            "execution_trace": state.execution_trace,
            "retrieved_chunks_count": len(state.retrieved_chunks),
            "retrieved_chunks": [c.model_dump() for c in state.retrieved_chunks],
        }

    @app.get("/api/v1/github/diff", tags=["github"])
    def get_github_diff(repo: str, commit: str) -> dict[str, str]:
        try:
            diff_text = github_client.fetch_commit_diff(repo, commit)
            return {"repository": repo, "commit_sha": commit, "diff": diff_text}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/v1/graph/entity", tags=["graph"])
    def add_graph_entity_endpoint(entity: GraphEntity) -> dict[str, str]:
        graph_store.add_entity(entity)
        return {"status": "success", "entity_id": entity.id}

    @app.post("/api/v1/graph/edge", tags=["graph"])
    def add_graph_edge_endpoint(edge: GraphEdge) -> dict[str, str]:
        graph_store.add_edge(edge)
        return {"status": "success"}

    @app.get("/api/v1/graph/traverse/{entity_id}", tags=["graph"])
    def traverse_graph_endpoint(entity_id: str, max_hops: int = 2) -> dict[str, object]:
        traversal = graph_store.multi_hop_query(entity_id, max_hops=max_hops)
        return {"start_entity_id": entity_id, "max_hops": max_hops, "traversal": traversal}

    @app.get("/api/v1/investigations", tags=["history"])
    def list_investigations(limit: int = 20) -> dict[str, object]:
        return {"items": store.list_investigations(limit=min(limit, 100))}

    @app.get(
        "/api/v1/investigations/{investigation_id}",
        response_model=DiagnoseResponse,
        tags=["history"],
    )
    def get_investigation(investigation_id: str) -> DiagnoseResponse:
        found = store.get_investigation(investigation_id)
        if found is None:
            raise HTTPException(status_code=404, detail="Investigation not found.")
        return found

    return app


app = create_app()