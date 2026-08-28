"""Primary research chat: Anthropic Messages, with tools.

Embeddings still come from local Nomic. This module only writes the answer
and decides which tools to call.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from anthropic import AnthropicError, AsyncAnthropic

from app.agent.tools import TOOL_SCHEMAS
from app.config import Settings


class AnthropicChatError(RuntimeError):
    pass


def anthropic_tools() -> list[dict]:
    converted = []
    for item in TOOL_SCHEMAS:
        function = item["function"]
        converted.append(
            {
                "name": function["name"],
                "description": function["description"],
                "input_schema": function["parameters"],
            }
        )
    return converted


def to_anthropic_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    system = ""
    converted: list[dict] = []

    for message in messages:
        role = message.get("role")
        content = message.get("content") or ""
        if role == "system":
            system = f"{system}\n\n{content}".strip()
            continue
        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id") or "",
                "content": content,
            }
            if (
                converted
                and converted[-1]["role"] == "user"
                and isinstance(converted[-1]["content"], list)
            ):
                converted[-1]["content"].append(block)
            else:
                converted.append({"role": "user", "content": [block]})
            continue
        if role == "assistant":
            blocks: list[dict] = []
            if content:
                blocks.append({"type": "text", "text": content})
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                raw = function.get("arguments") or "{}"
                try:
                    tool_input = json.loads(raw)
                except json.JSONDecodeError:
                    tool_input = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.get("id") or "",
                        "name": function.get("name") or "",
                        "input": tool_input if isinstance(tool_input, dict) else {},
                    }
                )
            if blocks:
                converted.append({"role": "assistant", "content": blocks})
            continue
        if role == "user":
            if (
                converted
                and converted[-1]["role"] == "user"
                and isinstance(converted[-1]["content"], str)
            ):
                converted[-1]["content"] = f"{converted[-1]['content']}\n\n{content}"
            elif (
                converted
                and converted[-1]["role"] == "user"
                and isinstance(converted[-1]["content"], list)
            ):
                converted[-1]["content"].append({"type": "text", "text": content})
            else:
                converted.append({"role": "user", "content": content})

    return system, converted


def _client(settings: Settings) -> AsyncAnthropic:
    if not settings.anthropic_api_key:
        raise AnthropicChatError(
            "ANTHROPIC_API_KEY is missing. Put it in backend/.env.local."
        )
    return AsyncAnthropic(api_key=settings.anthropic_api_key)


async def stream_agent_turn(
    settings: Settings,
    messages: list[dict],
    tools: list[dict],
    turn: dict,
) -> AsyncIterator[str]:
    """Stream one Anthropic turn. Tool-call turns stay silent; answers stream live."""
    del tools  # Anthropic uses its own schema; OpenAI tools are ignored here.
    turn["content"] = ""
    turn["tool_calls"] = []
    system, converted = to_anthropic_messages(messages)
    saw_tools = False

    try:
        client = _client(settings)
        async with client.messages.stream(
            model=settings.anthropic_model,
            max_tokens=2048,
            system=system or "You are a careful research assistant.",
            messages=converted,
            tools=anthropic_tools(),
        ) as stream:
            async for event in stream:
                event_type = getattr(event, "type", None)
                if event_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if block is not None and getattr(block, "type", None) == "tool_use":
                        saw_tools = True
                if event_type == "text" and not saw_tools:
                    text = getattr(event, "text", None) or ""
                    if text:
                        turn["content"] += text
                        yield text
            final = await stream.get_final_message()
    except AnthropicError as exc:
        raise AnthropicChatError(f"Anthropic chat failed: {exc}") from exc

    if getattr(final, "stop_reason", None) == "max_tokens":
        raise AnthropicChatError("Anthropic response reached its output limit.")

    calls = []
    for block in final.content:
        if getattr(block, "type", None) != "tool_use":
            continue
        calls.append(
            {
                "id": block.id,
                "name": block.name,
                "arguments": json.dumps(
                    block.input if isinstance(block.input, dict) else {}
                ),
            }
        )
    turn["tool_calls"] = calls


async def stream_final_answer(
    settings: Settings,
    messages: list[dict],
) -> AsyncIterator[str]:
    system, converted = to_anthropic_messages(messages)
    try:
        client = _client(settings)
        async with client.messages.stream(
            model=settings.anthropic_model,
            max_tokens=2048,
            system=system or "You are a careful research assistant.",
            messages=converted,
        ) as stream:
            async for text in stream.text_stream:
                if text:
                    yield text
            final = await stream.get_final_message()
    except AnthropicError as exc:
        raise AnthropicChatError(f"Anthropic stream failed: {exc}") from exc

    if getattr(final, "stop_reason", None) == "max_tokens":
        raise AnthropicChatError("Anthropic response reached its output limit.")
