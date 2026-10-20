"""BuildSleuth Streamlit Workbench â€” Stage S5 Verified Implementation."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import streamlit as st

# 1. Load server-side secrets into environment if configured in Streamlit
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    if "GITHUB_TOKEN" in st.secrets:
        os.environ["GITHUB_TOKEN"] = st.secrets["GITHUB_TOKEN"]
except Exception:
    pass

from app.agent.workflow import run_agentic_investigation
from app.config import get_settings
from app.diagnosis import diagnose_full
from app.graph.graph_store import GraphStore
from app.hybrid_rag import run_hybrid_retrieval
from app.models import DiagnoseRequest, RetrievalInspectRequest
from app.security.rate_limiter import SQLiteRateLimiter
from app.storage import InvestigationStore

st.set_page_config(
    page_title="BuildSleuth â€” CI Failure Intelligence",
    page_icon="ðŸ›¡ï¸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #020617; color: #f8fafc; }
    .stTextInput input, .stTextArea textarea { background-color: #0f172a; color: #38bdf8; border: 1px solid #1e293b; border-radius: 0.5rem; }
    .metric-card { background-color: #0f172a; border: 1px solid #1e293b; padding: 1.25rem; border-radius: 0.75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

settings = get_settings()
graph_store = GraphStore(settings.DATA_DIR / "graph.db")
investigation_store = InvestigationStore(settings.DATA_DIR / "buildsleuth.db")
rate_limiter = SQLiteRateLimiter(
    settings.DATA_DIR / "rate_limits.db", max_requests=15, window_seconds=900
)

if "investigation_history" not in st.session_state:
    st.session_state["investigation_history"] = []

st.sidebar.title("ðŸ›¡ï¸ BuildSleuth")
st.sidebar.caption("Evidence-Grounded CI Intelligence")

nav_mode = st.sidebar.radio(
    "Navigation",
    [
        "ðŸ” New Investigation",
        "ðŸ”Ž Retrieval Inspector",
        "ðŸŒ Multi-Hop Graph Explorer",
        "ðŸ“ˆ Evaluation Benchmark",
        "ðŸ—‚ï¸ Investigation History",
        "ðŸ“Š System & Config Status",
    ],
)

st.sidebar.markdown("---")
gemini_ready = bool(os.environ.get("GEMINI_API_KEY") or settings.GEMINI_API_KEY)
st.sidebar.markdown(
    f"**LLM Status**: {'ðŸŸ¢ Gemini API Configured' if gemini_ready else 'ðŸŸ¡ Deterministic Demo Mode (No API Key)'}"
)
st.sidebar.markdown(
    "**Architecture**: Hybrid RAG (Dense + BM25 + RRF) + Bounded Agent + GraphRAG + Citation Verifier."
)


def render_landing_and_form() -> None:
    st.title("BuildSleuth: CI Failure Intelligence Platform")
    st.markdown(
        "Diagnose failed software builds by connecting CI logs, source code, commit changes, "
        "and documentation into an explainable RAG pipeline with verified line citations."
    )

    if not gemini_ready:
        st.info(
            "â„¹ï¸ **Deterministic Demo Mode Active**: `GEMINI_API_KEY` is not configured. "
            "BuildSleuth will execute full Hybrid RAG (Dense + BM25 + RRF + Reranker) and deterministic "
            "root-cause synthesis with verified line citations."
        )

    with st.form("investigation_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            repo_input = st.text_input(
                "GitHub Repository (optional)",
                placeholder="owner/repo (e.g. PRANAY1836/BuildSleuth)",
            )
        with col2:
            use_gemini_toggle = st.checkbox(
                "Enable Gemini API Reasoning",
                value=True,
                help="Uses Google Gemini Developer API when GEMINI_API_KEY is configured; falls back safely otherwise.",
            )
        with col3:
            use_agent_toggle = st.checkbox(
                "Use Bounded Agentic Workflow",
                value=True,
                help="Runs evidence sufficiency checks and bounded query reformulation loops.",
            )

        question_input = st.text_input(
            "Investigation Question",
            value="Why did this build fail, what changed, and what should I fix next?",
        )

        sample_log_default = (
            "2026-10-20T10:00:01Z Run pytest\n"
            "collecting ... collected 1 item\n"
            "tests/test_app.py:1: in <module>\n"
            "    import pandas as lp\n"
            "E   ModuleNotFoundError: No module named 'pandas'\n"
            "Error: Process completed with exit code 1."
        )

        log_input = st.text_area(
            "Paste CI Build Log or Stack Trace",
            value=sample_log_default,
            height=200,
        )

        submitted = st.form_submit_button(
            "Run Evidence-Grounded Diagnosis", use_container_width=True
        )

    if submitted:
        raw_bytes = len(log_input.encode("utf-8"))
        if not log_input.strip():
            st.error("Please provide a non-empty build log or stack trace.")
            return
        if raw_bytes > settings.MAX_UPLOAD_BYTES:
            st.error(
                f"Log payload ({raw_bytes} bytes) exceeds maximum allowed limit ({settings.MAX_UPLOAD_BYTES} bytes)."
            )
            return

        client_bucket = "streamlit_public_demo_bucket"
        allowed, remaining = rate_limiter.check_rate_limit(client_bucket)
        if not allowed:
            st.error(
                "âš ï¸ Rate limit exceeded (Max 15 investigations per 15 minutes). Please wait before submitting again."
            )
            return

        with st.spinner(
            f"Executing Hybrid RAG & Evidence Verification... (Remaining window quota: {remaining})"
        ):
            try:
                t0 = time.perf_counter()
                request = DiagnoseRequest(
                    repository=repo_input.strip() if repo_input else None,
                    log_text=log_input,
                    question=question_input,
                    top_k=5,
                    use_gemini=use_gemini_toggle,
                )

                agent_trace = None
                if use_agent_toggle:
                    agent_result = run_agentic_investigation(log_text=log_input, question=question_input)
                    response = agent_result.diagnosis
                    agent_trace = agent_result
                else:
                    response = diagnose_full(request, settings=settings)

                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

                # Persist record in SQLite InvestigationStore
                try:
                    investigation_store.save_investigation(request, response)
                except Exception:
                    pass

                st.session_state["last_diagnosis"] = response
                st.session_state["last_agent_trace"] = agent_trace
                log_hash = hashlib.sha256(log_input.encode("utf-8")).hexdigest()[:10]
                st.session_state["investigation_history"].insert(
                    0,
                    {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "log_sha256_prefix": log_hash,
                        "failure_category": response.failure_category,
                        "confidence_level": response.confidence_level,
                        "diagnosis_mode": response.diagnosis_mode,
                        "latency_ms": elapsed_ms,
                        "summary": response.summary,
                    },
                )
                st.success(f"Investigation completed in {elapsed_ms} ms!")
            except Exception as exc:
                st.exception(exc)
                return

    if "last_diagnosis" in st.session_state:
        render_diagnosis_workspace(
            st.session_state["last_diagnosis"],
            st.session_state.get("last_agent_trace"),
        )


def render_diagnosis_workspace(res, agent_trace=None) -> None:
    st.markdown("---")
    st.header("ðŸ“‹ Diagnosis Workspace")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            f"<div class='metric-card'><b>Failure Category</b><br><span style='color: #38bdf8; font-size: 1.1rem;'>{res.failure_category}</span></div>",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f"<div class='metric-card'><b>Confidence Level</b><br><span style='color: #34d399; font-size: 1.1rem; text-transform: uppercase;'>{res.confidence_level}</span></div>",
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            f"<div class='metric-card'><b>Diagnosis Mode</b><br><span style='color: #f472b6; font-size: 1.1rem;'>{res.diagnosis_mode}</span></div>",
            unsafe_allow_html=True,
        )

    st.subheader("Summary")
    st.info(res.summary)

    st.subheader("Root Cause Analysis")
    st.markdown(
        f"<div style='background-color: #0f172a; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #38bdf8; font-family: monospace;'>{res.root_cause}</div>",
        unsafe_allow_html=True,
    )

    col_obs, col_rec = st.columns(2)
    with col_obs:
        st.markdown("#### Observed Facts")
        if res.observed_facts:
            for fact in res.observed_facts:
                st.markdown(f"- {fact}")
        else:
            st.caption("No explicit facts extracted.")

        st.markdown("#### Alternative Causes")
        if res.alternative_causes:
            for alt in res.alternative_causes:
                st.markdown(f"- {alt}")
        else:
            st.caption("None noted.")

    with col_rec:
        st.markdown("#### Recommended Actions")
        if res.recommended_actions:
            for act in res.recommended_actions:
                st.markdown(f"âœ… {act}")
        else:
            st.caption("No recommendations generated.")

        st.markdown("#### Missing Information & Limitations")
        for lim in res.limitations:
            st.markdown(f"âš ï¸ {lim}")

    if agent_trace is not None:
        st.markdown("---")
        st.subheader("ðŸ¤– Bounded Agent Execution Trace")
        st.caption(
            f"Iterations: {agent_trace.iterations} | Evidence Sufficient: `{agent_trace.evidence_sufficient}`"
        )
        for step in agent_trace.steps:
            st.markdown(
                f"- **Step {step.step_number} (`{step.node}`)**: {step.decision} *(Query: `{step.query_used}` | Top Score: {step.top_score:.3f})*"
            )

    st.markdown("---")
    st.subheader("ðŸ”Ž Retrieved Evidence & Verified Citations")
    if res.evidence_chunks:
        for chunk in res.evidence_chunks:
            with st.expander(
                f"[{chunk.source_id}] {chunk.source} (Score: {chunk.score:.3f})"
            ):
                st.code(chunk.text, language="text")
                st.caption(
                    f"Method: {chunk.retrieval_method} | Lines {chunk.line_start}-{chunk.line_end}"
                )
    else:
        st.caption("No evidence chunks retrieved.")

    if res.verification_report:
        v_rep = res.verification_report
        st.markdown("#### Citation Verification Audit")
        st.markdown(
            f"- **Verified Citations**: {v_rep.verified_citations} / {v_rep.total_citations} "
            f"(Precision: {v_rep.citation_precision * 100:.1f}%, Grounded: `{v_rep.all_verified}`)"
        )


def render_retrieval_inspector() -> None:
    st.title("ðŸ”Ž Hybrid Retrieval Inspector")
    st.markdown(
        "Inspect intermediate scores from Dense Semantic search, BM25 sparse matching, "
        "Reciprocal Rank Fusion (RRF), and Cross-Interaction Reranking."
    )

    with st.form("inspect_form"):
        query = st.text_input("Query", value="ModuleNotFoundError pandas")
        log_text = st.text_area(
            "Log Excerpt",
            value="Traceback:\n  File 'app/main.py', line 2\n    import pandas as lp\nModuleNotFoundError: No module named 'pandas'",
            height=150,
        )
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Top K Results", 1, 10, 5)
        with col2:
            rrf_k = st.number_input("RRF Constant (k)", value=60, min_value=1, max_value=200)

        run_inspect = st.form_submit_button("Run Hybrid Inspection", use_container_width=True)

    if run_inspect:
        with st.spinner("Executing Dense, BM25, RRF, and Reranking..."):
            req = RetrievalInspectRequest(
                log_text=log_text,
                query=query,
                top_k=top_k,
                rrf_k=int(rrf_k),
                enable_reranker=True,
            )
            inspection, _ = run_hybrid_retrieval(req)

        tab_dense, tab_bm25, tab_rrf, tab_rerank = st.tabs(
            ["Dense Semantic", "BM25 Sparse", "RRF Fused", "Final Reranked"]
        )

        with tab_dense:
            for c in inspection.dense_top:
                st.markdown(f"**Score**: `{c.dense_score:.4f}` | **Rank**: `#{c.dense_rank}`")
                st.code(c.text, language="text")

        with tab_bm25:
            for c in inspection.bm25_top:
                st.markdown(f"**Score**: `{c.bm25_score:.2f}` | **Rank**: `#{c.bm25_rank}`")
                st.code(c.text, language="text")

        with tab_rrf:
            for c in inspection.rrf_fused:
                st.markdown(
                    f"**RRF Score**: `{c.rrf_score:.6f}` | **Rank**: `#{c.rrf_rank}`"
                )
                st.code(c.text, language="text")

        with tab_rerank:
            for c in inspection.final_reranked:
                st.markdown(
                    f"**Rerank Score**: `{c.rerank_score:.4f}` | **Explanation**: {c.explanation}"
                )
                st.code(c.text, language="text")


def render_graph_explorer() -> None:
    st.title("ðŸŒ Multi-Hop GraphRAG Explorer")
    st.markdown(
        "Explore entity relationships connecting Repositories, Commits, Files, Modules, "
        "Dependencies, Jobs, and Failed Steps stored in SQLite."
    )

    with st.form("graph_query_form"):
        start_entity = st.text_input("Start Entity ID", value="repo_1")
        max_hops = st.slider("Max Traversal Hops", 1, 3, 2)
        query_graph = st.form_submit_button("Traverse Graph", use_container_width=True)

    if query_graph:
        traversal = graph_store.multi_hop_query(start_entity.strip(), max_hops=max_hops)
        st.subheader(f"Traversal Results from '{start_entity}' (Max Hops: {max_hops})")
        if traversal:
            for item in traversal:
                st.markdown(
                    f"- **Entity**: `{item['entity_name']}` ({item['entity_type']}) at **Hop {item['hops']}**"
                )
                st.json(item["path"])
        else:
            st.info("No connected entities found for this ID.")


def render_evaluation_benchmark() -> None:
    st.title("ðŸ“ˆ Multi-Metric Evaluation Benchmark")
    st.markdown(
        "Evaluate BuildSleuth against the labeled regression benchmark (`eval/dataset.json`). "
        "Measures Category Accuracy separately from Citation Precision, Abstention Accuracy, and Latency."
    )

    dataset_path = Path("eval/dataset.json")
    if not dataset_path.exists():
        st.error("eval/dataset.json not found in repository.")
        return

    raw_data = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = raw_data.get("cases", raw_data) if isinstance(raw_data, dict) else raw_data

    st.caption(f"Loaded **{len(cases)}** benchmark cases from `eval/dataset.json`.")

    if st.button("Run Benchmark Suite (Deterministic Mode)", use_container_width=True):
        results_rows = []
        correct_cat = 0
        citation_precisions = []
        abstention_total = 0
        abstention_correct = 0
        latencies = []

        for case in cases:
            case_id = case.get("id", "unknown")
            expected_cat = case.get("expected_category") or case.get("failure_category")
            log_text = case.get("log_text") or case.get("log", "")
            question = case.get("question", "Why did this build fail?")

            t0 = time.perf_counter()
            res = diagnose_full(
                DiagnoseRequest(log_text=log_text, question=question, use_gemini=False),
                settings=settings,
            )
            dt_ms = round((time.perf_counter() - t0) * 1000, 2)
            latencies.append(dt_ms)

            cat_match = res.failure_category == expected_cat
            if cat_match:
                correct_cat += 1

            cit_prec = (
                res.verification_report.citation_precision
                if res.verification_report
                else 1.0
            )
            citation_precisions.append(cit_prec)

            if expected_cat in ("unknown", "insufficient_evidence"):
                abstention_total += 1
                if res.confidence_level in ("low", "insufficient_evidence"):
                    abstention_correct += 1

            results_rows.append(
                {
                    "Case ID": case_id,
                    "Expected": expected_cat,
                    "Predicted": res.failure_category,
                    "Match": "âœ…" if cat_match else "âŒ",
                    "Confidence": res.confidence_level,
                    "Citation Precision": f"{cit_prec * 100:.0f}%",
                    "Latency (ms)": dt_ms,
                }
            )

        total = max(1, len(cases))
        mean_cit = (sum(citation_precisions) / len(citation_precisions)) * 100 if citation_precisions else 0.0
        mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
        abst_rate = (abstention_correct / abstention_total * 100) if abstention_total else 100.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Category Accuracy", f"{(correct_cat / total) * 100:.1f}% ({correct_cat}/{total})")
        c2.metric("Mean Citation Precision", f"{mean_cit:.1f}%")
        c3.metric("Abstention Accuracy", f"{abst_rate:.1f}%")
        c4.metric("Mean Latency", f"{mean_lat:.2f} ms")

        st.dataframe(results_rows, use_container_width=True)


def render_history_page() -> None:
    st.title("ðŸ—‚ï¸ Investigation History")
    st.markdown(
        "Audit log of investigations executed in this session and persisted in SQLite (`data/buildsleuth.db`). "
        "Note: On Streamlit Community Cloud, local SQLite storage is ephemeral across container restarts."
    )
    history = st.session_state.get("investigation_history", [])
    if history:
        st.subheader("Current Session Runs")
        st.dataframe(history, use_container_width=True)

    try:
        persisted = investigation_store.list_recent(limit=20)
        if persisted:
            st.subheader("SQLite Persisted Investigations")
            st.dataframe(persisted, use_container_width=True)
    except Exception:
        pass

    if not history:
        st.info("No investigations run in this session yet.")


def render_status_page() -> None:
    st.title("System & Integration Status")
    summary = settings.safe_diagnostic_summary()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Environment & Limits")
        st.json(
            {
                "app_env": summary["app_env"],
                "app_version": summary["app_version"],
                "max_upload_bytes": summary["max_upload_bytes"],
                "rate_limit_rpm": summary["rate_limit_rpm"],
            }
        )
    with col2:
        st.subheader("Features & Integrations")
        st.json(summary["features"])
        st.json(summary["integrations"])


if nav_mode == "ðŸ” New Investigation":
    render_landing_and_form()
elif nav_mode == "ðŸ”Ž Retrieval Inspector":
    render_retrieval_inspector()
elif nav_mode == "ðŸŒ Multi-Hop Graph Explorer":
    render_graph_explorer()
elif nav_mode == "ðŸ“ˆ Evaluation Benchmark":
    render_evaluation_benchmark()
elif nav_mode == "ðŸ—‚ï¸ Investigation History":
    render_history_page()
else:
    render_status_page()

