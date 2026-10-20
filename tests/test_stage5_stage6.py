"""Stage 5 (Agentic workflow self-correction) and Stage 6 (Graph store multi-hop traversal) tests."""

from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from app.agent.workflow import run_agentic_investigation
from app.config import Settings
from app.graph.graph_store import GraphEdge, GraphEntity, GraphStore
from app.main import create_app


def get_client(tmp_path: Path) -> TestClient:
    settings = Settings(
        APP_ENV="test",
        DATA_DIR=tmp_path,
        SQLITE_DB_PATH=tmp_path / "test_stage5_6.db",
    )
    return TestClient(create_app(settings_override=settings))


def test_agent_workflow_self_correction_loop() -> None:
    log_text = "ModuleNotFoundError: No module named 'pydantic'"
    state = run_agentic_investigation(
        log_text=log_text,
        question="Why did build fail?",
        max_retries=2,
    )
    assert state.sufficiency == "sufficient"
    assert len(state.execution_trace) >= 2
    assert any("retrieval_node" in step for step in state.execution_trace)


def test_graph_store_multi_hop_traversal(tmp_path: Path) -> None:
    store = GraphStore(tmp_path / "graph.db")

    # Add entities: Repo -> File -> Dependency
    store.add_entity(GraphEntity(id="repo_1", entity_type="Repository", name="BuildSleuth"))
    store.add_entity(GraphEntity(id="file_1", entity_type="File", name="app/main.py"))
    store.add_entity(GraphEntity(id="dep_1", entity_type="Dependency", name="fastapi"))

    store.add_edge(GraphEdge(source_id="repo_1", target_id="file_1", relation="contains", provenance="observed"))
    store.add_edge(GraphEdge(source_id="file_1", target_id="dep_1", relation="imports", provenance="observed"))

    results = store.multi_hop_query("repo_1", max_hops=2)
    assert len(results) == 2
    target_names = {r["entity_name"] for r in results}
    assert "app/main.py" in target_names
    assert "fastapi" in target_names


def test_agent_and_graph_api_endpoints(tmp_path: Path) -> None:
    client = get_client(tmp_path)

    # Test Agent Investigate API
    agent_resp = client.post(
        "/api/v1/agent/investigate",
        json={"log_text": "AssertionError: test failed"},
    )
    assert agent_resp.status_code == 200
    assert "execution_trace" in agent_resp.json()

    # Test Graph Entity and Traverse APIs
    ent_resp = client.post(
        "/api/v1/graph/entity",
        json={"id": "test_job", "entity_type": "Job", "name": "CI Test Run"},
    )
    assert ent_resp.status_code == 200

    trav_resp = client.get("/api/v1/graph/traverse/test_job?max_hops=1")
    assert trav_resp.status_code == 200
    assert trav_resp.json()["start_entity_id"] == "test_job"