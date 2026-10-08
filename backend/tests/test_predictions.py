from io import BytesIO
import sys
from pathlib import Path

from fastapi.testclient import TestClient


BACKEND_DIR = Path(__file__).resolve().parent.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from main import app


client = TestClient(app)

CSV_CONTENT = """HR,O2Sat,Temp,SBP,MAP,Resp,WBC,Creatinine,Glucose,Age,ICULOS
85,98,37.0,120,80,18,8.5,1.0,100,45,10
90,97,37.2,118,78,19,9.0,1.1,105,45,11
95,96,37.5,115,75,21,10.2,1.2,110,45,12
"""


def test_prediction_endpoint_exists():
    paths = {
        route.path
        for route in app.routes
        if hasattr(route, "path")
    }

    assert "/patients/{patient_id}/predict" in paths


def test_prediction_for_nonexistent_patient():
    response = client.post(
        "/patients/999999/predict",
        files={
            "file": (
                "vitals.csv",
                BytesIO(CSV_CONTENT.encode()),
                "text/csv",
            )
        },
    )

    assert response.status_code == 404


def test_prediction_requires_file():
    response = client.post("/patients/999999/predict")

    assert response.status_code in (400, 404, 422)


def test_prediction_rejects_invalid_csv():
    response = client.post(
        "/patients/999999/predict",
        files={
            "file": (
                "invalid.csv",
                BytesIO(b"this,is,not,valid,vitals,data"),
                "text/csv",
            )
        },
    )

    assert response.status_code in (400, 404, 422)


def test_prediction_response_structure(monkeypatch):
    from routes import predictions

    monkeypatch.setattr(
        predictions.database,
        "get_patient_by_id",
        lambda patient_id: {
            "id": patient_id,
            "name": "Test Patient",
            "bed_number": "B-01",
            "age": 45,
            "gender": "M",
            "created_at": "2026-10-08T10:00:00",
        },
    )

    monkeypatch.setattr(
        predictions.database,
        "save_prediction",
        lambda **kwargs: 101,
    )

    monkeypatch.setattr(
        predictions.database,
        "save_vitals_batch",
        lambda patient_id, rows: None,
    )

    monkeypatch.setattr(
        predictions.database,
        "get_latest_prediction",
        lambda patient_id: None,
    )

    monkeypatch.setattr(
        predictions.predictor,
        "predict",
        lambda df: {
            "probability": 0.82,
            "risk_level": "HIGH",
            "features": {
                "HR": 95,
                "O2Sat": 96,
                "Temp": 37.5,
            },
        },
    )

    monkeypatch.setattr(
        predictions.shap_explainer,
        "explain",
        lambda df: [
            {
                "feature": "HR",
                "impact": 0.23,
                "direction": "increases_risk",
            }
        ],
    )

    monkeypatch.setattr(
        predictions.predictor,
        "clinical_recommendation",
        lambda risk_level: "Immediate clinical review recommended.",
    )

    response = client.post(
        "/patients/1/predict",
        files={
            "file": (
                "vitals.csv",
                BytesIO(CSV_CONTENT.encode()),
                "text/csv",
            )
        },
    )

    assert response.status_code == 200, response.text

    data = response.json()

    assert data["patient_id"] == 1
    assert "sepsis_probability" in data
    assert "risk_level" in data
    assert "shap_factors" in data
    assert "predicted_at" in data
    assert "model_version" in data
    assert "clinical_recommendation" in data
    assert "prediction_id" in data

    assert 0 <= data["sepsis_probability"] <= 1
    assert data["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert isinstance(data["shap_factors"], list)
    assert data["prediction_id"] == 101