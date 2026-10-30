from fastapi.testclient import TestClient

from api.main import app
from src.predict import load_model

client = TestClient(app)
FEATURES = load_model()["features"]


def valid_transaction():
    return {"features": {name: 0.0 for name in FEATURES}}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_transaction_returns_prediction():
    response = client.post("/predict", json=valid_transaction())
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"is_fraud", "risk_score", "threshold"}
    assert 0.0 <= body["risk_score"] <= 1.0


def test_missing_feature_returns_422():
    payload = valid_transaction()
    del payload["features"]["V5"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "V5" in response.json()["detail"]


def test_negative_amount_returns_422():
    payload = valid_transaction()
    payload["features"]["Amount"] = -10.0
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_non_numeric_value_returns_422():
    payload = valid_transaction()
    payload["features"]["V1"] = "abc"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422