import asyncio

import numpy as np
from fastapi.testclient import TestClient

from app.agent.loop import RESEARCH_INTERRUPTED, RESEARCH_UNAVAILABLE
from app.main import app
from app.store.memory import Chunk


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


def test_failed_embedding_preserves_previous_corpus(monkeypatch):
    with TestClient(app) as client:
        app.state.store.add(
            [Chunk("previous.txt", "previous valid source", 0)],
            np.array([[1.0, 0.0]], dtype=np.float32),
        )

        async def fail_embedding(_chunks):
            raise RuntimeError("internal embedding detail")

        monkeypatch.setattr(app.state.embed_queue, "embed", fail_embedding)
        response = client.post(
            "/api/sources",
            files=[("files", ("replacement.txt", b"new source", "text/plain"))],
        )

        assert response.status_code == 502
        assert (
            response.text
            == "Source processing failed. Existing sources were preserved."
        )
        assert [chunk.file_name for chunk in app.state.store.chunks] == ["previous.txt"]


def test_multi_file_upload_embeds_files_concurrently(monkeypatch):
    state = {"active": 0, "max_active": 0}

    async def tracked_embedding(chunks):
        state["active"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        await asyncio.sleep(0.01)
        state["active"] -= 1
        return np.ones((len(chunks), 2), dtype=np.float32)

    with TestClient(app) as client:
        monkeypatch.setattr(app.state.embed_queue, "embed", tracked_embedding)
        response = client.post(
            "/api/sources",
            files=[
                ("files", ("one.txt", b"first source", "text/plain")),
                ("files", ("two.txt", b"second source", "text/plain")),
                ("files", ("three.txt", b"third source", "text/plain")),
            ],
        )

        assert response.status_code == 200
        assert state["max_active"] == 3
        assert {chunk.file_name for chunk in app.state.store.chunks} == {
            "one.txt",
            "two.txt",
            "three.txt",
        }


def test_research_failure_before_streaming_returns_non_200(monkeypatch):
    async def unavailable(*_args, **_kwargs):
        from app.agent.loop import ResearchUnavailableError

        raise ResearchUnavailableError("internal provider detail")
        yield ""

    monkeypatch.setattr("app.routes.research.run_research", unavailable)
    with TestClient(app) as client:
        response = client.post("/api/research", json={"request": "test failure"})

    assert response.status_code == 502
    assert response.text == RESEARCH_UNAVAILABLE
    assert "provider" not in response.text.lower()


def test_research_failure_after_streaming_is_sanitized(monkeypatch):
    async def interrupted(*_args, **_kwargs):
        from app.agent.loop import ResearchInterruptedError

        yield "partial answer"
        raise ResearchInterruptedError("internal provider detail")

    monkeypatch.setattr("app.routes.research.run_research", interrupted)
    with TestClient(app) as client:
        response = client.post("/api/research", json={"request": "test interruption"})

    assert response.status_code == 200
    assert response.text == f"partial answer\n\n> {RESEARCH_INTERRUPTED}"
    assert "provider" not in response.text.lower()
