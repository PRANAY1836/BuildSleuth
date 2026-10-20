"""Stage 3 (Hybrid RAG, RRF, Metadata Filters, Reranking) & Stage 4 (Gemini + Citation Verification) tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config import Settings
from app.gemini_client import GeminiStructuredOutput
from app.main import create_app
from app.models import CitationItem, EvidenceChunk
from app.verification import verify_citations


def get_client(tmp_path: Path, gemini_key: str = "") -> TestClient:
    settings = Settings(
        APP_ENV="test",
        DATA_DIR=tmp_path,
        SQLITE_DB_PATH=tmp_path / "test_stage3_4.db",
        GEMINI_API_KEY=gemini_key,
    )
    return TestClient(create_app(settings_override=settings))


def test_retrieval_inspector_exposes_dense_bm25_rrf_and_rerank(tmp_path: Path) -> None:
    client = get_client(tmp_path)
    sample_log = (
        "Step 1: Installing dependencies\n"
        "Collecting pytest>=8.0\n"
        "Step 2: Running unit tests in tests/test_auth.py\n"
        "E   AssertionError: expected status 200 but got 401\n"
        "FAILED tests/test_auth.py::test_login_token\n"
    )
    resp = client.post(
        "/api/v1/retrieval/inspect",
        json={
            "log_text": sample_log,
            "query": "Why did test_login_token fail in test_auth.py?",
            "diff": "--- a/app/auth.py\n+++ b/app/auth.py\n@@ -10,1 +10,1 @@\n-return 200\n+return 401",
            "repository": "PRANAY1836/BuildSleuth",
            "top_k": 4,
            "rrf_k": 60,
            "enable_reranker": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_chunks_indexed"] >= 3
    assert data["reranker_applied"] is True
    assert len(data["dense_top"]) >= 1
    assert len(data["bm25_top"]) >= 1
    assert len(data["rrf_fused"]) >= 1
    assert len(data["final_reranked"]) >= 1

    top_chunk = data["final_reranked"][0]
    assert top_chunk["rrf_score"] > 0.0
    assert top_chunk["rerank_score"] > 0.0
    assert "RRF=#" in top_chunk["explanation"]


def test_metadata_filtering_in_hybrid_rag(tmp_path: Path) -> None:
    client = get_client(tmp_path)
    sample_log = "ModuleNotFoundError: No module named 'httpx'\n"
    resp = client.post(
        "/api/v1/retrieval/inspect",
        json={
            "log_text": sample_log,
            "query": "missing module httpx",
            "diff": "+import httpx",
            "filters": {"source_types": ["commit_diff"]},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert all(c["source_type"] == "commit_diff" for c in data["final_reranked"])


def test_citation_verifier_catches_hallucinated_ids_and_lines() -> None:
    real_chunk = EvidenceChunk(
        source_id="chk_ci_log_1_5_abcdef123456",
        source_type="ci_log",
        source="ci_log:L1-L5",
        text="File 'app/main.py', line 4\nModuleNotFoundError: No module named 'redis'",
        line_start=1,
        line_end=5,
    )

    citations = [
        # Valid citation
        CitationItem(
            claim="Missing redis module reported on line 2.",
            source_id="chk_ci_log_1_5_abcdef123456",
            line_start=1,
            line_end=5,
            quoted_Anchor="No module named 'redis'",
        ),
        # Invalid source_id
        CitationItem(
            claim="Fake citation to non-existent chunk.",
            source_id="chk_fake_999",
            line_start=1,
            line_end=2,
        ),
        # Out-of-bounds line numbers
        CitationItem(
            claim="Out of bounds line reference.",
            source_id="chk_ci_log_1_5_abcdef123456",
            line_start=1,
            line_end=50,
        ),
    ]

    report = verify_citations(citations, [real_chunk])
    assert report.total_citations == 3
    assert report.verified_citations == 1
    assert report.all_verified is False
    assert "chk_fake_999" in report.invalid_source_ids
    assert len(report.out_of_bounds_lines) == 1


def test_mocked_gemini_diagnosis_with_verified_citations(tmp_path: Path) -> None:
    client = get_client(tmp_path, gemini_key="AIzaSyMOCK_KEY_FOR_UNIT_TEST_12345678")
    sample_log = "E   ModuleNotFoundError: No module named 'fastapi'\nError: exit code 1"

    # First inspect to get the deterministic chunk ID
    inspect_resp = client.post(
        "/api/v1/retrieval/inspect",
        json={"log_text": sample_log, "query": "Why did it fail?"},
    ).json()
    real_chunk = next(
        c for c in inspect_resp["final_reranked"] if c["source_type"] == "ci_log"
    )

    mock_gemini_output = GeminiStructuredOutput(
        failure_category="missing_dependency",
        summary="Missing fastapi package in CI environment.",
        root_cause="The `fastapi` dependency is not installed before running tests.",
        is_inferred=False,
        confidence_level="high",
        confidence_explanation="Directly stated in the traceback.",
        observed_facts=["ModuleNotFoundError for 'fastapi'."],
        alternative_causes=["Wrong virtualenv activated."],
        recommended_actions=["Add fastapi to requirements.txt."],
        missing_information=[],
        citations=[
            CitationItem(
                claim="Log reports missing fastapi module.",
                source_id=real_chunk["source_id"],
                line_start=real_chunk["line_start"],
                line_end=real_chunk["line_end"],
                quoted_Anchor="No module named 'fastapi'",
            )
        ],
    )

    with patch(
        "app.gemini_client.generate_gemini_diagnosis",
        return_value=(mock_gemini_output, None),
    ):
        resp = client.post("/api/v1/diagnose", json={"log_text": sample_log})
        assert resp.status_code == 200
        body = resp.json()
        assert body["diagnosis_mode"] == "hybrid_rag_gemini"
        assert body["failure_category"] == "missing_dependency"
        assert body["verification_report"]["all_verified"] is True
        assert body["verification_report"]["verified_citations"] == 1