from fastapi.testclient import TestClient

from app.main import app


def test_research_rejects_empty_request():
    with TestClient(app) as client:
        response = client.post("/api/research", json={"request": "   "})
    assert response.status_code == 400


def test_sources_rejects_empty_file():
    with TestClient(app) as client:
        response = client.post(
            "/api/sources",
            files=[("files", ("empty.txt", b"", "text/plain"))],
        )
    assert response.status_code == 400
