"""Pydantic request and response schemas for BuildSleuth."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, model_validator


class Evidence(BaseModel):
    """Starter-compatible evidence item."""

    source: str
    snippet: str
    score: float


class EvidenceChunk(BaseModel):
    """Line-numbered, deterministically identified evidence chunk with hybrid RAG scores."""

    source_id: str = Field(..., description="Stable deterministic SHA-256 chunk ID.")
    source_type: Literal["ci_log", "source_file", "commit_diff", "dependency_manifest", "doc"] = (
        "ci_log"
    )
    source: str = Field(default="ci_log", description="Human-readable source path or label.")
    text: str = Field(..., description="Exact normalized text snippet.")
    line_start: int = Field(default=1, ge=1)
    line_end: int = Field(default=1, ge=1)
    timestamp_start: str | None = None
    repository: str | None = None
    commit_sha: str | None = None
    file_path: str | None = None
    failure_type: str | None = None

    # Final and intermediate retrieval scores for explainability
    score: float = 0.0
    dense_score: float = 0.0
    dense_rank: int | None = None
    bm25_score: float = 0.0
    bm25_rank: int | None = None
    rrf_score: float = 0.0
    rrf_rank: int | None = None
    rerank_score: float = 0.0
    retrieval_method: str = "hybrid_rrf_rerank"
    explanation: str = ""


class CandidateCause(BaseModel):
    """Starter-compatible candidate root cause hypothesis."""

    title: str
    rationale: str
    recommended_checks: list[str]
    confidence: str = "heuristic"


class CitationItem(BaseModel):
    """Explicit link between a diagnosis claim and a retrieved EvidenceChunk."""

    claim: str
    source_id: str
    line_start: int = Field(default=1, ge=1)
    line_end: int = Field(default=1, ge=1)
    quoted_Anchor: str = ""


class CitationVerificationReport(BaseModel):
    """Deterministic verification audit of model-generated citations."""

    total_citations: int = 0
    verified_citations: int = 0
    citation_precision: float = 1.0
    invalid_source_ids: list[str] = Field(default_factory=list)
    out_of_bounds_lines: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    all_verified: bool = True


class RetrievalFilter(BaseModel):
    """Optional metadata filters for hybrid retrieval."""

    repository: str | None = None
    commit_sha: str | None = None
    source_types: list[str] | None = None
    file_path_prefix: str | None = None
    failure_type: str | None = None


class RetrievalInspectRequest(BaseModel):
    """Request payload for inspecting intermediate hybrid RAG rankings."""

    log_text: str = Field(..., min_length=1, max_length=5_000_000)
    query: str = Field(
        default="Why did this build fail, what changed, and what should I fix next?",
        min_length=2,
        max_length=1000,
    )
    diff: str | None = Field(default=None, max_length=500_000)
    repository: str | None = None
    commit_sha: str | None = None
    filters: RetrievalFilter | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    rrf_k: int = Field(default=60, ge=1, le=200)
    enable_reranker: bool = True


class RetrievalInspectResponse(BaseModel):
    """Detailed breakdown of Dense, BM25, RRF, and Reranked retrieval stages."""

    query: str
    total_chunks_indexed: int
    filtered_chunks_count: int
    rrf_k: int
    reranker_applied: bool
    embedding_backend: str
    dense_top: list[EvidenceChunk]
    bm25_top: list[EvidenceChunk]
    rrf_fused: list[EvidenceChunk]
    final_reranked: list[EvidenceChunk]


class DiagnoseRequest(BaseModel):
    """Input payload supporting both starter `log` and extended `log_text` fields."""

    repository: str | None = Field(default=None, max_length=300)
    run_id: str | None = Field(default=None, max_length=200)
    commit_sha: str | None = Field(default=None, max_length=64)
    log: str | None = Field(default=None, max_length=5_000_000)
    log_text: str | None = Field(default=None, max_length=5_000_000)
    diff: str | None = Field(default=None, max_length=500_000)
    question: str = Field(
        default="Why did this build fail, what changed, and what should I fix next?",
        min_length=3,
        max_length=1000,
    )
    top_k: int = Field(default=5, ge=1, le=20)
    use_gemini: bool = Field(
        default=True,
        description="Attempt Gemini structured diagnosis when API key is configured.",
    )

    @model_validator(mode="after")
    def validate_log_input(self) -> DiagnoseRequest:
        resolved = self.log if self.log is not None else self.log_text
        if resolved is None or not resolved.strip():
            raise ValueError("Either 'log' or 'log_text' must be provided and non-empty.")
        self.log = resolved
        self.log_text = resolved
        return self


class DiagnoseResponse(BaseModel):
    """Unified response satisfying starter tests and full evidence-grounded diagnosis."""

    investigation_id: str | None = None
    repository: str | None = None
    run_id: str | None = None
    commit_sha: str | None = None

    # Starter-compatible fields
    extracted_signals: list[str] = Field(default_factory=list)
    candidate_causes: list[CandidateCause] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    # Rich Stage 1–4 diagnosis fields
    failure_category: str = "unknown_or_insufficient_evidence"
    summary: str = ""
    root_cause: str = ""
    is_inferred: bool = True
    confidence_level: Literal["high", "medium", "low", "insufficient_evidence"] = (
        "insufficient_evidence"
    )
    confidence_explanation: str = ""
    observed_facts: list[str] = Field(default_factory=list)
    alternative_causes: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    citations: list[CitationItem] = Field(default_factory=list)
    verification_report: CitationVerificationReport = Field(
        default_factory=CitationVerificationReport
    )
    evidence_chunks: list[EvidenceChunk] = Field(default_factory=list)
    prompt_injection_detected: bool = False
    diagnosis_mode: Literal["heuristic_baseline", "hybrid_rag_heuristic", "hybrid_rag_gemini"] = (
        "hybrid_rag_heuristic"
    )


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "buildsleuth"
    version: str = "0.2.0"


class ReadinessResponse(BaseModel):
    status: Literal["ready", "degraded"]
    checks: dict[str, bool]
    details: dict[str, object]