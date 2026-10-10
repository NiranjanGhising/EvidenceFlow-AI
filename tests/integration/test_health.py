from fastapi.testclient import TestClient

from evidenceflow.api.app import app


def test_health_endpoint() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "EvidenceFlow AI"
