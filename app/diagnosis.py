"""CI failure diagnosis engine integrating Hybrid RAG, Gemini reasoning, and Citation Verification."""

from __future__ import annotations

import re

from app import gemini_client
from app.config import Settings, get_settings
from app.hybrid_rag import run_hybrid_retrieval
from app.models import (
    CandidateCause,
    CitationItem,
    DiagnoseRequest,
    DiagnoseResponse,
    Evidence,
    EvidenceChunk,
    RetrievalInspectRequest,
)
from app.retrieval import retrieve
from app.verification import verify_citations

RULES: list[tuple[re.Pattern[str], str, str, str, list[str]]] = [
    (
        re.compile(r"ModuleNotFoundError|No module named|ImportError", re.I),
        "missing_dependency",
        "Missing or unresolved Python dependency",
        "The log contains a Python import error.",
        [
            "Check the missing package is declared in project dependencies.",
            "Confirm the CI job installs dependencies in the active environment.",
            "Check Python and package version compatibility.",
        ],
    ),
    (
        re.compile(
            r"ResolutionImpossible|Could not find a version that satisfies|dependency conflict",
            re.I,
        ),
        "dependency_conflict",
        "Dependency resolution conflict",
        "The log indicates package constraints could not be resolved.",
        [
            "Inspect dependency and lockfile changes.",
            "Verify runtime and package version compatibility.",
            "Reproduce the install command in a clean environment.",
        ],
    ),
    (
        re.compile(r"SyntaxError:\s*(.+)", re.I),
        "syntax_error",
        "Syntax or parsing error",
        "The log contains a language syntax error.",
        [
            "Inspect the exact file and line number preceding the SyntaxError.",
            "Confirm the CI Python runtime version supports the syntax used.",
        ],
    ),
    (
        re.compile(r"AssertionError|assert .*==|FAILED .*test", re.I),
        "test_failure",
        "Failing test or changed behavior",
        "The log contains a test failure signal.",
        [
            "Start with the first failing assertion and compare expected versus actual output.",
            "Inspect changed code and relevant fixtures.",
            "Rerun the single failing test with verbose output.",
        ],
    ),
    (
        re.compile(
            r"KeyError|missing environment variable|environment variable .* not set", re.I
        ),
        "configuration_error",
        "Missing key or configuration",
        "The log suggests a missing key or required configuration value.",
        [
            "Check required configuration names in CI settings.",
            "Verify secrets are configured without printing their values.",
            "Compare local and CI environment assumptions.",
        ],
    ),
    (
        re.compile(
            r"timed out|timeout|ConnectionError|Temporary failure in name resolution", re.I
        ),
        "network_or_service_failure",
        "Network, availability, or timeout issue",
        "The log contains a timeout or connectivity signal.",
        [
            "Check external service availability and network access.",
            "Review timeout and retry settings.",
            "Determine whether the test depends on a flaky external service.",
        ],
    ),
    (
        re.compile(r"flake8|ruff|black|eslint|prettier|lint", re.I),
        "lint_or_format_failure",
        "Linting or formatting failure",
        "The log references a linter or formatter.",
        [
            "Run the same tool version locally.",
            "Inspect reported files and line numbers.",
            "Check that CI and local configuration are aligned.",
        ],
    ),
]


def diagnose(
    log: str, diff: str | None = None
) -> tuple[list[str], list[CandidateCause], list[Evidence]]:
    """Starter-compatible diagnosis function returning (signals, causes, evidence)."""
    signals: list[str] = []
    causes: list[CandidateCause] = []
    for pattern, _cat, title, rationale, checks in RULES:
        matches = pattern.findall(log)
        if matches:
            signals.append(title)
            causes.append(
                CandidateCause(title=title, rationale=rationale, recommended_checks=checks)
            )
    if diff and diff.strip():
        signals.append("Code diff supplied")
    context = log + "\n" + (diff or "")
    ranked = retrieve(context, limit=3)
    evidence = [
        Evidence(source=doc.source, snippet=doc.text, score=round(score, 3))
        for doc, score in ranked
    ]
    if not causes:
        causes.append(
            CandidateCause(
                title="No known failure signature matched",
                rationale=(
                    "The baseline rules did not recognize a common failure signature; "
                    "this is not evidence that the log has no root cause."
                ),
                recommended_checks=[
                    "Inspect the earliest error in the log rather than only the final summary.",
                    "Include the failing step, relevant diff, and runtime details.",
                    "Add a regression fixture if this failure pattern recurs.",
                ],
            )
        )
    return signals, causes, evidence


def _cite_chunk(chunk: EvidenceChunk | None) -> str:
    if chunk is None:
        return "[unreferenced log region]"
    return f"[{chunk.source_id} L{chunk.line_start}-L{chunk.line_end}]"


def diagnose_full(
    request: DiagnoseRequest,
    settings: Settings | None = None,
) -> DiagnoseResponse:
    """Execute Hybrid RAG retrieval, optional Gemini reasoning, and deterministic citation verification."""
    cfg = settings or get_settings()
    raw_log = request.log or request.log_text or ""
    signals, causes, starter_evidence = diagnose(raw_log, request.diff)

    inspect_req = RetrievalInspectRequest(
        log_text=raw_log,
        query=request.question,
        diff=request.diff,
        repository=request.repository,
        commit_sha=request.commit_sha,
        top_k=request.top_k,
        rrf_k=cfg.RRF_K,
        enable_reranker=cfg.ENABLE_RERANKER,
    )
    hybrid_result, injection_detected = run_hybrid_retrieval(inspect_req)
    chunks = hybrid_result.final_reranked

    if not starter_evidence and chunks:
        starter_evidence = [
            Evidence(source=c.source, snippet=c.text, score=c.score) for c in chunks[:3]
        ]

    limitations: list[str] = []
    if injection_detected:
        limitations.append(
            "WARNING: Untrusted prompt-injection phrasing was detected in the input log and isolated inside <untrusted_evidence> delimiters."
        )

    if request.use_gemini and cfg.gemini_configured and chunks:
        gemini_out, gemini_err = gemini_client.generate_gemini_diagnosis(
            question=request.question,
            chunks=chunks,
            settings=cfg,
        )
        if gemini_out is not None:
            v_report = verify_citations(gemini_out.citations, chunks)
            if not v_report.all_verified:
                limitations.append(
                    f"Citation verifier flagged {len(v_report.unsupported_claims)} claim(s) with unverified citations."
                )
            return DiagnoseResponse(
                repository=request.repository,
                run_id=request.run_id,
                commit_sha=request.commit_sha,
                extracted_signals=signals,
                candidate_causes=causes,
                evidence=starter_evidence,
                limitations=limitations
                or ["Generated via Hybrid RAG + Gemini structured diagnosis with verified citations."],
                failure_category=gemini_out.failure_category,
                summary=gemini_out.summary,
                root_cause=gemini_out.root_cause,
                is_inferred=gemini_out.is_inferred,
                confidence_level=gemini_out.confidence_level,
                confidence_explanation=gemini_out.confidence_explanation,
                observed_facts=gemini_out.observed_facts,
                alternative_causes=gemini_out.alternative_causes,
                recommended_actions=gemini_out.recommended_actions,
                missing_information=gemini_out.missing_information,
                citations=gemini_out.citations,
                verification_report=v_report,
                evidence_chunks=chunks,
                prompt_injection_detected=injection_detected,
                diagnosis_mode="hybrid_rag_gemini",
            )
        if gemini_err:
            limitations.append(gemini_err)
    else:
        limitations.append(
            "Diagnosis generated using Hybrid RAG (Dense + BM25 + RRF + Reranker) and deterministic rule verification."
        )

    matched_category = "unknown_or_insufficient_evidence"
    observed_facts: list[str] = []
    citations: list[CitationItem] = []
    primary_chunk = next(
        (c for c in chunks if c.source_type == "ci_log"), chunks[0] if chunks else None
    )

    for pattern, category, title, _rationale, _checks in RULES:
        m = pattern.search(raw_log)
        if m:
            if matched_category == "unknown_or_insufficient_evidence":
                matched_category = category
            citing_chunk = next(
                (c for c in chunks if pattern.search(c.text)), primary_chunk
            )
            matched_str = m.group(0).strip()
            fact_text = f"Detected '{title}' signature (`{matched_str}`) at {_cite_chunk(citing_chunk)}."
            observed_facts.append(fact_text)
            if citing_chunk is not None:
                m_in_chunk = pattern.search(citing_chunk.text)
                anchor = m_in_chunk.group(0) if m_in_chunk else ""
                citations.append(
                    CitationItem(
                        claim=fact_text,
                        source_id=citing_chunk.source_id,
                        line_start=citing_chunk.line_start,
                        line_end=citing_chunk.line_end,
                        quoted_Anchor=anchor,
                    )
                )

    v_report = verify_citations(citations, chunks)

    if matched_category != "unknown_or_insufficient_evidence":
        top_cause = causes[0]
        citation_label = _cite_chunk(primary_chunk)
        return DiagnoseResponse(
            repository=request.repository,
            run_id=request.run_id,
            commit_sha=request.commit_sha,
            extracted_signals=signals,
            candidate_causes=causes,
            evidence=starter_evidence,
            limitations=limitations,
            failure_category=matched_category,
            summary=f"{top_cause.title}: {top_cause.rationale}",
            root_cause=f"{top_cause.title} evidenced in {citation_label}.",
            is_inferred=False,
            confidence_level="high",
            confidence_explanation=f"Direct failure signature matched and ranked #1 by Hybrid RRF in {citation_label}.",
            observed_facts=observed_facts,
            alternative_causes=[c.title for c in causes[1:]]
            or ["Stale CI cache or environment mismatch between local and runner."],
            recommended_actions=top_cause.recommended_checks,
            missing_information=(
                ["Commit diff (`diff`) and dependency lockfile (`requirements.txt` / `pyproject.toml`)."]
                if not request.diff
                else []
            ),
            citations=citations,
            verification_report=v_report,
            evidence_chunks=chunks,
            prompt_injection_detected=injection_detected,
            diagnosis_mode="hybrid_rag_heuristic",
        )

    return DiagnoseResponse(
        repository=request.repository,
        run_id=request.run_id,
        commit_sha=request.commit_sha,
        extracted_signals=signals,
        candidate_causes=causes,
        evidence=starter_evidence,
        limitations=limitations,
        failure_category="unknown_or_insufficient_evidence",
        summary="No known failure signature matched in the provided log excerpt.",
        root_cause=(
            "Insufficient evidence to determine a specific root cause without guessing. "
            "Inspect the earliest non-zero exit step or supply the commit diff."
        ),
        is_inferred=True,
        confidence_level="insufficient_evidence",
        confidence_explanation="No terminal exception pattern matched; abstaining from fabricating a root cause.",
        observed_facts=observed_facts,
        alternative_causes=[
            "The root error occurred in an earlier build step not included in this log snippet.",
            "External runner termination (timeout or out-of-memory signal).",
        ],
        recommended_actions=causes[0].recommended_checks if causes else [],
        missing_information=["Full step log around exit code and recent git diff."],
        citations=citations,
        verification_report=v_report,
        evidence_chunks=chunks,
        prompt_injection_detected=injection_detected,
        diagnosis_mode="hybrid_rag_heuristic",
    )   