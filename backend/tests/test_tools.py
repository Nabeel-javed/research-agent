import numpy as np
import pytest

from app.agent.tools import search_uploads
from app.config import Settings
from app.store.memory import Chunk, MemoryStore


@pytest.mark.asyncio
async def test_search_returns_complete_chunk_without_preview_truncation(monkeypatch):
    important_tail = "Northstar is the current employer."
    chunk_text = "A" * 600 + "\n" + important_tail
    store = MemoryStore()
    store.add(
        [Chunk("Nabeel-CV.pdf", chunk_text, 0)],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )

    async def fake_embed(_settings, _query):
        return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr("app.agent.tools.embed_query", fake_embed)

    result = await search_uploads(Settings(), store, "current employer")

    assert important_tail in result
    assert "[Source: Nabeel-CV.pdf | chunk: 0" in result
    assert "A" * 600 in result


@pytest.mark.asyncio
async def test_search_preserves_chunk_structure(monkeypatch):
    store = MemoryStore()
    store.add(
        [Chunk("resume.txt", "Experience\nNorthstar\nLead Engineer", 0)],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )

    async def fake_embed(_settings, _query):
        return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr("app.agent.tools.embed_query", fake_embed)

    result = await search_uploads(Settings(), store, "Northstar")

    assert "Experience\nNorthstar\nLead Engineer" in result
