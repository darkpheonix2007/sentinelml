import pytest
from fastapi.testclient import TestClient

from app import api


@pytest.fixture
def client(monkeypatch, monitor):
    monkeypatch.setattr(api, "monitor", lambda: monitor)
    monkeypatch.setattr(api, "Store", lambda: monitor.store)
    monkeypatch.setattr(api, "load_bundle", lambda: monitor.bundle)
    return TestClient(api.app)


def test_api_round_trip(client, frame):
    assert client.get("/health").status_code == 200
    payload = {"batch_id": "api-test", "records": frame.to_dict("records")}
    result = client.post("/batches", json=payload)
    assert result.status_code == 201, result.text
    assert client.post("/batches", json=payload).status_code == 409
    assert client.get("/batches/api-test/metrics").json()["report"]["labels_available"] is False
    labels = {str(i): int(v) for i, v in enumerate(frame.PAY_0 > 0)}
    result = client.post("/batches/api-test/labels", json={"labels": labels})
    assert result.status_code == 200, result.text
    assert result.json()["label_status"] == "complete"
    assert client.get("/batches/no-such-batch").status_code == 404
    assert client.get("/alerts").status_code == 200
    assert client.get("/alerts/no-such-alert").status_code == 404


def test_api_invalid_input(client, frame):
    assert client.post("/batches", json={"records": []}).status_code == 422
    assert client.post("/predict", json={"records": [{"AGE": 25}]}).status_code == 422
    result = client.post("/predict", json={"records": frame.head(2).to_dict("records")})
    assert result.status_code == 200
    assert len(result.json()["probabilities"]) == 2
