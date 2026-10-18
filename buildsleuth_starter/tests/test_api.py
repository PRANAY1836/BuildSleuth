from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_diagnoses_missing_dependency():
    response = client.post("/diagnose", json={
        "repository": "demo/service",
        "run_id": "local-001",
        "log": "ModuleNotFoundError: No module named 'requests'",
    })
    assert response.status_code == 200
    data = response.json()
    assert any("dependency" in item.lower() for item in data["extracted_signals"])
    assert data["candidate_causes"]
    assert data["limitations"]


def test_unknown_failure_does_not_claim_root_cause():
    response = client.post("/diagnose", json={"log": "step terminated with code 74"})
    assert response.status_code == 200
    data = response.json()
    assert data["candidate_causes"][0]["title"] == "No known failure signature matched"


def test_rejects_empty_log():
    response = client.post("/diagnose", json={"log": ""})
    assert response.status_code == 422
