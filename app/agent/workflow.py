"""Stage 5: Bounded agentic orchestration state machine with self-correction and query reformulation."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

from app.config import get_settings
from app.hybrid_rag import run_hybrid_retrieval
from app.models import EvidenceChunk, RetrievalInspectRequest


class AgentState(BaseModel):
    """Explicit state schema for the investigation agent workflow."""

    query: str
    original_query: str
    log_text: str
    diff: str | None = None
    repository: str | None = None
    commit_sha: str | None = None
    retrieved_chunks: list[EvidenceChunk] = Field(default_factory=list)
    sufficiency: Literal["sufficient", "insufficient", "needs_reformulation"] = "insufficient"
    reformulation_count: int = 0
    max_retries: int = 2
    execution_trace: list[str] = Field(default_factory=list)


def classify_and_route_node(state: AgentState) -> AgentState:
    """Node 1: Classify query and determine initial retrieval strategy."""
    state.execution_trace.append("classify_and_route: Query analyzed and routed to Hybrid RAG.")
    return state


def retrieval_node(state: AgentState) -> AgentState:
    """Node 2: Execute hybrid RAG retrieval."""
    settings = get_settings()
    inspect_req = RetrievalInspectRequest(
        log_text=state.log_text,
        query=state.query,
        diff=state.diff,
        repository=state.repository,
        commit_sha=state.commit_sha,
        top_k=5,
        rrf_k=settings.RRF_K,
        enable_reranker=settings.ENABLE_RERANKER,
    )
    result, _ = run_hybrid_retrieval(inspect_req)
    state.retrieved_chunks = result.final_reranked
    state.execution_trace.append(
        f"retrieval_node: Retrieved {len(state.retrieved_chunks)} evidence chunks."
    )
    return state


def sufficiency_check_node(state: AgentState) -> AgentState:
    """Node 3: Assess whether retrieved evidence is sufficient to diagnose the failure."""
    top_score = state.retrieved_chunks[0].score if state.retrieved_chunks else 0.0
    has_error_keyword = any(
        kw in state.log_text.lower()
        for kw in ["error", "failed", "exception", "traceback", "modulenotfounderror"]
    )

    if top_score > 0.35 or (not has_error_keyword and len(state.retrieved_chunks) > 0):
        state.sufficiency = "sufficient"
        state.execution_trace.append(
            f"sufficiency_check: Evidence sufficient (top_score={top_score:.3f})."
        )
    elif state.reformulation_count < state.max_retries:
        state.sufficiency = "needs_reformulation"
        state.execution_trace.append(
            f"sufficiency_check: Evidence insufficient (top_score={top_score:.3f}); triggering reformulation (attempt {state.reformulation_count + 1})."
        )
    else:
        state.sufficiency = "sufficient"  # Max retries reached; proceed with available evidence
        state.execution_trace.append(
            "sufficiency_check: Max reformulation retries reached; proceeding with best available evidence."
        )
    return state


def query_reformulation_node(state: AgentState) -> AgentState:
    """Node 4: Bounded query reformulation adding error signifiers."""
    state.reformulation_count += 1
    state.query = f"{state.original_query} traceback error signature exit code failed exception"
    state.execution_trace.append(f"query_reformulation: Reformulated query to '{state.query}'.")
    return state


def run_agentic_investigation(
    log_text: str,
    question: str,
    diff: str | None = None,
    repository: str | None = None,
    commit_sha: str | None = None,
    max_retries: int = 2,
) -> AgentState:
    """Execute the bounded agentic diagnosis loop."""
    state = AgentState(
        query=question,
        original_query=question,
        log_text=log_text,
        diff=diff,
        repository=repository,
        commit_sha=commit_sha,
        max_retries=max_retries,
    )

    state = classify_and_route_node(state)
    for _step in range(max_retries + 1):
        state = retrieval_node(state)
        state = sufficiency_check_node(state)
        if state.sufficiency == "sufficient":
            break
        if state.sufficiency == "needs_reformulation":
            state = query_reformulation_node(state)

    state.execution_trace.append("agent_workflow: Investigation pipeline successfully completed.")
    return state