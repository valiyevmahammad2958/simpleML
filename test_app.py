from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_valid_prediction():
    response = client.post("/predict", json={"size_m2": 120})

    assert response.status_code == 200
    assert "predicted_price" in response.json()


def test_negative_house_size():
    response = client.post("/predict", json={"size_m2": -100})

    assert response.status_code == 422