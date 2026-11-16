from fastapi.testclient import TestClient

from hardened_llm_gateway.app import create_app


def test_health() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
