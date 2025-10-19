# tests/test_api_endpoints.py
import os
import shutil
from pathlib import Path

import pytest

# 1) Train once so artifacts/model.joblib exists BEFORE importing app
@pytest.fixture(scope="session", autouse=True)
def trained_model(tmp_path_factory):
    # If your train entrypoint is src.train:main
    from src.train import main  # noqa
    # Ensure a clean artifacts dir at repo root (as api.py expects)
    artifacts = Path("artifacts")
    if artifacts.exists():
        shutil.rmtree(artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)

    # Set version env if your train.py reads it; else ignored
    os.environ["MODEL_VERSION"] = "v0.1"
    main()
    assert (artifacts / "model.joblib").exists()

@pytest.fixture
def client():
    # Import AFTER training so api can load the model successfully
    from fastapi.testclient import TestClient
    from src.api import app
    return TestClient(app)

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_version" in body

def test_predict_ok(client):
    payload = {
        "age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03,
        "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], float)

def test_predict_bad_input_returns_json_error(client):
    r = client.post("/predict", json={"age": "not-a-float"})
    # Pydantic validation will reject and FastAPI returns 422 Unprocessable Entity
    assert r.status_code in (400, 422)
