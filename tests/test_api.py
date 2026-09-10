import pytest
from fastapi.testclient import TestClient
from src.api.server import app

API_KEY = "dev-api-key-change-in-production"
AUTH_HEADERS = {"X-API-Key": API_KEY}


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict_requires_auth(client):
    payload = {
        "age": 30,
        "bmi": 25.0,
        "children": 0,
        "smoker": "no",
        "region": "southwest"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 403


def test_model_info(client):
    response = client.get("/model-info", headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "metrics" in data
    assert "coverage_80" in data["metrics"]


def test_predict_single_applicant(client):
    payload = {
        "age": 30,
        "bmi": 25.0,
        "children": 0,
        "smoker": "no",
        "region": "southwest"
    }
    response = client.post("/predict", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    
    assert "predicted_cost" in data
    assert "interval_80" in data
    assert len(data["interval_80"]) == 2
    assert data["interval_80"][0] <= data["predicted_cost"] <= data["interval_80"][1]
    assert data["premium_tier"] in ["Standard", "Loaded", "Refer to Underwriter"]
    assert "reason_codes" in data
    assert len(data["reason_codes"]) <= 3
    assert "decision_timestamp" in data


def test_predict_high_risk_tier(client):
    # Older smoker with high BMI -> should be loaded or refer to underwriter
    payload = {
        "age": 60,
        "bmi": 38.5,
        "children": 1,
        "smoker": "yes",
        "region": "southeast"
    }
    response = client.post("/predict", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["premium_tier"] in ["Loaded", "Refer to Underwriter"]
    assert data["predicted_cost"] > 16640.0


def test_predict_invalid_input_validation(client):
    # Invalid age (> 100)
    payload = {
        "age": 150,
        "bmi": 25.0,
        "children": 0,
        "smoker": "no",
        "region": "southwest"
    }
    response = client.post("/predict", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 422


def test_batch_prediction(client):
    payload = {
        "applicants": [
            {"age": 25, "bmi": 22.0, "children": 0, "smoker": "no", "region": "northeast"},
            {"age": 55, "bmi": 32.0, "children": 2, "smoker": "yes", "region": "southeast"}
        ]
    }
    response = client.post("/predict/batch", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2


def test_drift_with_payload(client):
    payload = {
        "applicants": [
            {"age": 30, "bmi": 25.0, "children": 0, "smoker": "no", "region": "southwest"},
            {"age": 35, "bmi": 28.0, "children": 1, "smoker": "no", "region": "northwest"}
        ]
    }
    response = client.post("/drift", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "drift_metrics" in data
    assert "age" in data["drift_metrics"]
    assert "bmi" in data["drift_metrics"]
    assert "categorical_drift" in data
    assert "smoker" in data["categorical_drift"]
    assert data["sample_size"] == 2


def test_audit_logs_endpoint(client):
    # Make a prediction to ensure at least one log exists
    payload = {
        "age": 40,
        "bmi": 27.5,
        "children": 1,
        "smoker": "no",
        "region": "northwest"
    }
    client.post("/predict", json=payload, headers=AUTH_HEADERS)
    
    response = client.get("/audit-logs?limit=10", headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["total_logged"] >= 1
    assert len(data["records"]) >= 1
    record = data["records"][0]
    assert "predicted_cost" in record
    assert "premium_tier" in record



