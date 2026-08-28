"""OpenAI-compatible client for the local LM Studio server."""

from __future__ import annotations

from collections.abc import AsyncIterator

from openai import APIError, AsyncOpenAI

from app.config import Settings
from app.llm.think import ThinkStripper


class LmStudioError(RuntimeError):
    pass


def _client(settings: Settings) -> AsyncOpenAI:
    if not settings.lm_studio_api_token:
        raise LmStudioError(
            "LM_STUDIO_API_TOKEN is missing. Put it in backend/.env.local."
        )
    return AsyncOpenAI(
        base_url=settings.lm_studio_base_url,
        api_key=settings.lm_studio_api_token,
        timeout=120.0,
    )


async def embed_texts(settings: Settings, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    try:
        client = _client(settings)
        response = await client.embeddings.create(
            model=settings.embed_model, input=texts
        )
    except APIError as exc:
        raise LmStudioError(f"LM Studio embeddings failed: {exc}") from exc
    return [item.embedding for item in response.data]


async def chat_once(
    settings: Settings,
    messages: list[dict],
    tools: list[dict] | None = None,
) -> dict:
    try:
        client = _client(settings)
        response = await client.chat.completions.create(
            model=settings.chat_model,
            messages=messages,
            tools=tools or None,
            temperature=0.2,
        )
    except APIError as exc:
        raise LmStudioError(f"LM Studio chat failed: {exc}") from exc

    choice = response.choices[0]
    message = choice.message
    tool_calls = []
    for call in message.tool_calls or []:
        tool_calls.append(
            {
                "id": call.id,
                "name": call.function.name,
                "arguments": call.function.arguments,
            }
        )
    content = message.content or ""
    stripper = ThinkStripper()
    content = stripper.feed(content) + stripper.flush()
    return {"content": content, "tool_calls": tool_calls}


async def stream_agent_turn(
    settings: Settings,
    messages: list[dict],
    tools: list[dict],
    turn: dict,
) -> AsyncIterator[str]:
    """Stream one chat turn. Tool-call turns stay silent; answers stream live.

    `turn` is filled with `content` and `tool_calls` when the iterator ends.
    If the model starts a tool call, visible tokens are not forwarded so the
    UI does not print function JSON.
    """
    turn["content"] = ""
    turn["tool_calls"] = []
    stripper = ThinkStripper()
    calls_by_index: dict[int, dict] = {}
    saw_tools = False
    finish_reason = None

    try:
        client = _client(settings)
        stream = await client.chat.completions.create(
            model=settings.chat_model,
            messages=messages,
            tools=tools,
            temperature=0.2,
            stream=True,
        )
        async for event in stream:
            choice = event.choices[0] if event.choices else None
            if not choice or not choice.delta:
                continue
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta
            if delta.tool_calls:
                saw_tools = True
                for call in delta.tool_calls:
                    index = call.index if call.index is not None else 0
                    slot = calls_by_index.setdefault(
                        index, {"id": "", "name": "", "arguments": ""}
                    )
                    if call.id:
                        slot["id"] = call.id
                    function = call.function
                    if function:
                        if function.name:
                            slot["name"] += function.name
                        if function.arguments:
                            slot["arguments"] += function.arguments
            if saw_tools or not delta.content:
                continue
            visible = stripper.feed(delta.content)
            if visible:
                turn["content"] += visible
                yield visible
        if not saw_tools:
            leftover = stripper.flush()
            if leftover:
                turn["content"] += leftover
                yield leftover
    except APIError as exc:
        raise LmStudioError(f"LM Studio chat stream failed: {exc}") from exc

    if finish_reason == "length":
        raise LmStudioError("LM Studio response reached its output limit.")

    turn["tool_calls"] = [calls_by_index[index] for index in sorted(calls_by_index)]


async def stream_final_answer(
    settings: Settings,
    messages: list[dict],
) -> AsyncIterator[str]:
    stripper = ThinkStripper()
    finish_reason = None
    try:
        client = _client(settings)
        stream = await client.chat.completions.create(
            model=settings.chat_model,
            messages=messages,
            temperature=0.2,
            stream=True,
        )
        async for event in stream:
            choice = event.choices[0] if event.choices else None
            if choice and choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta.content if choice else None
            if not delta:
                continue
            visible = stripper.feed(delta)
            if visible:
                yield visible
        leftover = stripper.flush()
        if leftover:
            yield leftover
    except APIError as exc:
        raise LmStudioError(f"LM Studio stream failed: {exc}") from exc

    if finish_reason == "length":
        raise LmStudioError("LM Studio response reached its output limit.")
