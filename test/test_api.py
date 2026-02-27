from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.post(
        "/predict",
        json={"observation": [10, 0.1, 100, 5, 20]}
    )

    assert response.status_code == 200
    data = response.json()

    assert "mitigation_action" in data
    assert 0 <= data["mitigation_action"] <= 4
