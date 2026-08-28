from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from app.agent.loop import EMPTY_ANSWER, run_research
from app.store.memory import MemoryStore


class FakeSettings(SimpleNamespace):
    max_agent_steps: int = 5


async def _collect(stream: AsyncIterator[str]) -> str:
    parts: list[str] = []
    async for piece in stream:
        parts.append(piece)
    return "".join(parts)


@pytest.mark.asyncio
async def test_anthropic_answer_is_forwarded_without_calling_local_qwen(monkeypatch):
    async def fake_turn(_settings, _messages, _tools, turn):
        turn["content"] = ""
        turn["tool_calls"] = []
        yield "Hello "
        yield "world"

    async def local_must_not_run(*_args, **_kwargs):
        raise AssertionError("Local Qwen must not run when Anthropic succeeds")
        yield ""

    monkeypatch.setattr("app.agent.loop.stream_agent_turn", fake_turn)
    monkeypatch.setattr("app.agent.loop.qwen_agent_turn", local_must_not_run)

    text = await _collect(run_research(FakeSettings(), MemoryStore(), "What is this?"))
    assert text == "Hello world"


@pytest.mark.asyncio
async def test_local_qwen_runs_if_anthropic_returns_empty(monkeypatch):
    async def empty_anthropic(_settings, _messages, _tools, turn):
        turn["content"] = ""
        turn["tool_calls"] = []
        if False:
            yield ""

    async def local_turn(_settings, _messages, _tools, turn):
        turn["content"] = ""
        turn["tool_calls"] = []
        yield "local answer"

    monkeypatch.setattr("app.agent.loop.stream_agent_turn", empty_anthropic)
    monkeypatch.setattr("app.agent.loop.qwen_agent_turn", local_turn)

    text = await _collect(run_research(FakeSettings(), MemoryStore(), "What is this?"))
    assert "Falling back to the local Qwen model" in text
    assert text.endswith("local answer")


@pytest.mark.asyncio
async def test_both_providers_empty_gets_a_visible_message(monkeypatch):
    async def empty_turn(_settings, _messages, _tools, turn):
        turn["content"] = ""
        turn["tool_calls"] = []
        if False:
            yield ""

    monkeypatch.setattr("app.agent.loop.stream_agent_turn", empty_turn)
    monkeypatch.setattr("app.agent.loop.qwen_agent_turn", empty_turn)

    text = await _collect(run_research(FakeSettings(), MemoryStore(), "What is this?"))
    assert "Falling back to the local Qwen model" in text
    assert text.endswith(EMPTY_ANSWER)


@pytest.mark.asyncio
async def test_tool_turn_stays_off_stream_then_streams_answer(monkeypatch):
    calls = {"n": 0}

    async def fake_turn(_settings, _messages, _tools, turn):
        calls["n"] += 1
        if calls["n"] == 1:
            turn["content"] = ""
            turn["tool_calls"] = [
                {"id": "call-1", "name": "search_uploads", "arguments": '{"query":"cv"}'}
            ]
            return
            yield ""  # make this an async generator
        turn["content"] = ""
        turn["tool_calls"] = []
        yield "Cited from resume"

    async def fake_tool(_settings, _store, name, _arguments):
        assert name == "search_uploads"
        return "chunk text"

    monkeypatch.setattr("app.agent.loop.stream_agent_turn", fake_turn)
    monkeypatch.setattr("app.agent.loop.run_tool", fake_tool)

    text = await _collect(run_research(FakeSettings(), MemoryStore(), "Summarise the CV"))
    assert text == "Cited from resume"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_local_qwen_runs_if_anthropic_fails(monkeypatch):
    from app.llm.anthropic_chat import AnthropicChatError

    async def broken_anthropic(*_args, **_kwargs):
        raise AnthropicChatError("no cloud")
        yield ""  # keep this an async generator

    async def local_turn(_settings, _messages, _tools, turn):
        turn["content"] = ""
        turn["tool_calls"] = []
        yield "local answer"

    monkeypatch.setattr("app.agent.loop.stream_agent_turn", broken_anthropic)
    monkeypatch.setattr("app.agent.loop.qwen_agent_turn", local_turn)

    text = await _collect(run_research(FakeSettings(), MemoryStore(), "What is this?"))
    assert "Falling back to the local Qwen model" in text
    assert text.endswith("local answer")
