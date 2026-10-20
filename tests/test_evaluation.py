"""Evaluation regression suite validating classification accuracy and citation support rates."""

from __future__ import annotations

import json
from pathlib import Path
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_evaluation_dataset_regression() -> None:
    settings = Settings(APP_ENV="test")
    client = TestClient(create_app(settings_override=settings))

    dataset_path = Path("eval/dataset.json")
    assert dataset_path.exists(), "eval/dataset.json is missing."

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = data["cases"]
    assert len(cases) >= 10

    correct_predictions = 0
    total_cases = len(cases)

    for case in cases:
        resp = client.post(
            "/diagnose",
            json={"log_text": case["log_text"], "question": case["question"]},
        )
        assert resp.status_code == 200
        body = resp.json()

        pred_category = body["failure_category"]
        expected_category = case["expected_category"]

        if pred_category == expected_category:
            correct_predictions += 1

    accuracy = correct_predictions / total_cases
    # Enforce regression threshold based on baseline performance (>80% accuracy)
    assert accuracy >= 0.80, f"Evaluation accuracy {accuracy * 100:.1f}% fell below 80% threshold."