"""API Integration Test Suite.

Validates:
1. GET /api/health returns success.
2. GET /api/models returns the three available models.
3. POST /api/predict/lung-cancer accepts valid input and returns expected envelope.
4. POST /api/predict/asthma accepts valid input and returns expected envelope.
5. POST /api/predict/parkinsons accepts valid input and returns expected envelope.
6. Missing request body is rejected with 400.
7. Missing required features are rejected with 400.
8. Invalid request format (non-JSON) is rejected with 400.
9. Prediction does not retrain models or modify model artifacts (mtime check).
10. Model loading failure is handled safely with a controlled 500 error.
11. Unknown endpoint returns 404.
12. Disallowed HTTP method returns 405.
"""

import json
import os
import sys
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.api.app import create_app
from backend.services.prediction_service import PredictionService, get_prediction_service

LUNG_CANCER_SAMPLE = {
    "gender": "MALE",
    "age": 65,
    "smoking": 1,
    "yellow_fingers": 1,
    "anxiety": 1,
    "peer_pressure": 0,
    "chronic_disease": 1,
    "fatigue": 1,
    "allergy": 1,
    "wheezing": 1,
    "alcohol_consuming": 1,
    "coughing": 1,
    "shortness_of_breath": 1,
    "swallowing_difficulty": 0,
    "chest_pain": 1,
}

ASTHMA_SAMPLE = {
    "Age": 45,
    "Gender": "Female",
    "BMI": 28.5,
    "Smoking_Status": "Former",
    "Family_History": 1,
    "Allergies": "Pollen",
    "Air_Pollution_Level": "High",
    "Physical_Activity_Level": "Sedentary",
    "Occupation_Type": "Indoor",
    "Comorbidities": "None",
    "Medication_Adherence": 0.65,
    "Number_of_ER_Visits": 1,
    "Peak_Expiratory_Flow": 350.0,
    "FeNO_Level": 35.0,
}

PARKINSONS_SAMPLE = {
    "MDVP:Fo(Hz)": 119.992,
    "MDVP:Fhi(Hz)": 157.302,
    "MDVP:Flo(Hz)": 74.997,
    "MDVP:Jitter(%)": 0.00784,
    "MDVP:Jitter(Abs)": 0.00007,
    "MDVP:RAP": 0.0037,
    "MDVP:PPQ": 0.00554,
    "Jitter:DDP": 0.01109,
    "MDVP:Shimmer": 0.04374,
    "MDVP:Shimmer(dB)": 0.426,
    "Shimmer:APQ3": 0.02182,
    "Shimmer:APQ5": 0.0313,
    "MDVP:APQ": 0.02971,
    "Shimmer:DDA": 0.06545,
    "NHR": 0.02211,
    "HNR": 21.033,
    "RPDE": 0.414783,
    "DFA": 0.815285,
    "spread1": -4.813031,
    "spread2": 0.266482,
    "D2": 2.301442,
    "PPE": 0.284654,
}


@pytest.fixture
def client():
    """Create test client fixture."""
    app = create_app({"TESTING": True})
    with app.test_client() as client:
        yield client


def test_01_api_health(client):
    """Verify GET /api/health returns 200 and expected status."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert data["service"] == "ai-multi-disease-risk-screening-api"


def test_02_api_models_availability(client):
    """Verify GET /api/models returns all three available disease models."""
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.get_json()
    assert "models" in data
    models = data["models"]
    assert len(models) == 3

    disease_names = {m["disease"] for m in models}
    assert disease_names == {"lung_cancer", "asthma", "parkinsons"}
    for m in models:
        assert m["available"] is True


def test_03_predict_lung_cancer_valid(client):
    """Verify POST /api/predict/lung-cancer processes valid inputs."""
    res = client.post("/api/predict/lung-cancer", json=LUNG_CANCER_SAMPLE)
    assert res.status_code == 200
    data = res.get_json()
    assert data["disease"] == "lung_cancer"
    assert "prediction" in data
    pred = data["prediction"]
    assert pred["prediction"] in ("YES", "NO")
    assert pred["class"] in (0, 1)
    assert 0.0 <= pred["probability"] <= 1.0
    assert pred["risk_level"] in ("Low", "Moderate", "High")
    assert "disclaimer" in pred


def test_04_predict_asthma_valid(client):
    """Verify POST /api/predict/asthma processes valid inputs."""
    res = client.post("/api/predict/asthma", json=ASTHMA_SAMPLE)
    assert res.status_code == 200
    data = res.get_json()
    assert data["disease"] == "asthma"
    assert "prediction" in data
    pred = data["prediction"]
    assert pred["prediction"] in ("Has Asthma", "No Asthma")
    assert pred["class"] in (0, 1)
    assert 0.0 <= pred["probability"] <= 1.0
    assert pred["risk_level"] in ("Low", "Moderate", "High")
    assert "disclaimer" in pred


def test_05_predict_parkinsons_valid(client):
    """Verify POST /api/predict/parkinsons processes valid inputs."""
    res = client.post("/api/predict/parkinsons", json=PARKINSONS_SAMPLE)
    assert res.status_code == 200
    data = res.get_json()
    assert data["disease"] == "parkinsons"
    assert "prediction" in data
    pred = data["prediction"]
    assert pred["prediction"] in ("Parkinson's Disease", "Healthy / No Parkinson's Detected")
    assert pred["class"] in (0, 1)
    assert 0.0 <= pred["probability"] <= 1.0
    assert pred["risk_level"] in ("Low", "Moderate", "High")
    assert "disclaimer" in pred


def test_06_missing_request_body_rejected(client):
    """Verify missing request body or empty payload is rejected with 400."""
    res = client.post("/api/predict/lung-cancer", json={})
    assert res.status_code == 400
    data = res.get_json()
    assert "error" in data
    assert data["error"]["code"] == "INVALID_INPUT"


def test_07_missing_required_features_rejected(client):
    """Verify partial feature payload is rejected with 400."""
    incomplete = dict(LUNG_CANCER_SAMPLE)
    del incomplete["age"]
    del incomplete["smoking"]
    res = client.post("/api/predict/lung-cancer", json=incomplete)
    assert res.status_code == 400
    data = res.get_json()
    assert "error" in data
    assert data["error"]["code"] == "INVALID_INPUT"


def test_08_invalid_request_format_rejected(client):
    """Verify non-JSON payload is rejected with 400."""
    res = client.post(
        "/api/predict/lung-cancer",
        data="not a json string",
        content_type="text/plain",
    )
    assert res.status_code == 400
    data = res.get_json()
    assert data["error"]["code"] == "INVALID_REQUEST"


def test_09_prediction_does_not_retrain_or_mutate_artifacts(client):
    """Verify that predictions do not retrain models or touch artifact modification times."""
    parkinsons_model_path = os.path.join(REPO_ROOT, "backend", "models", "parkinsons", "model.joblib")
    mtime_before = os.path.getmtime(parkinsons_model_path)

    res = client.post("/api/predict/parkinsons", json=PARKINSONS_SAMPLE)
    assert res.status_code == 200

    mtime_after = os.path.getmtime(parkinsons_model_path)
    assert mtime_before == mtime_after, "Model artifact was modified during prediction!"


def test_10_model_loading_failure_handled_safely(client, monkeypatch):
    """Verify missing model artifacts return a safe 500 error without exposing stack traces."""
    service = get_prediction_service()

    def mock_get_module(name):
        raise FileNotFoundError(f"Simulated missing model for {name}")

    monkeypatch.setattr(service, "_get_module", mock_get_module)

    res = client.post("/api/predict/lung-cancer", json=LUNG_CANCER_SAMPLE)
    assert res.status_code == 500
    data = res.get_json()
    assert data["error"]["code"] == "MODEL_UNAVAILABLE"
    # Ensure no internal filesystem paths leaked to client
    assert "C:\\" not in data["error"]["message"]
    assert "P:\\" not in data["error"]["message"]


def test_11_unknown_route_returns_404(client):
    """Verify unknown endpoint returns clean 404 JSON response."""
    res = client.get("/api/unknown-endpoint")
    assert res.status_code == 404
    data = res.get_json()
    assert data["error"]["code"] == "NOT_FOUND"


def test_12_method_not_allowed_returns_405(client):
    """Verify disallowed HTTP verb returns clean 405 JSON response."""
    res = client.get("/api/predict/lung-cancer")
    assert res.status_code == 405
    data = res.get_json()
    assert data["error"]["code"] == "METHOD_NOT_ALLOWED"
