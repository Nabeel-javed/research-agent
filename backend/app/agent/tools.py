"""Retrieval and web-search tools available to the research agent."""

from __future__ import annotations

import json

import httpx
import numpy as np

from app.config import Settings
from app.llm.lm_studio import embed_texts
from app.store.memory import MemoryStore

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_uploads",
            "description": "Search the user's uploaded source files for relevant passages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for in the uploads."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the public web with Brave. Use when uploads are not enough.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Web search query."}
                },
                "required": ["query"],
            },
        },
    },
]


async def search_uploads(settings: Settings, store: MemoryStore, query: str) -> str:
    vectors = await embed_texts(settings, [query])
    hits = store.search(np.asarray(vectors[0], dtype=np.float32), settings.retrieve_top_k)
    if not hits:
        return "No uploaded sources matched this query. The corpus may be empty."

    passages = []
    for chunk, score in hits:
        passages.append(
            f"[Source: {chunk.file_name} | chunk: {chunk.index} | relevance: {score:.3f}]\n"
            f"{chunk.text}"
        )
    return "\n\n".join(passages)


async def web_search(settings: Settings, query: str) -> str:
    if not settings.brave_api_key:
        return "Brave search is not configured (BRAVE_API_KEY missing)."

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": 5},
                headers={
                    "X-Subscription-Token": settings.brave_api_key,
                    "Accept": "application/json",
                },
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        return f"Brave search failed: {exc}"

    results = payload.get("web", {}).get("results", [])
    if not results:
        return "Brave returned no web results."

    lines = []
    for item in results[:5]:
        title = item.get("title") or "untitled"
        url = item.get("url") or ""
        snippet = (item.get("description") or "").replace("\n", " ")
        lines.append(f"- {title}\n  {url}\n  {snippet}")
    return "\n".join(lines)


async def run_tool(
    settings: Settings,
    store: MemoryStore,
    name: str,
    raw_arguments: str,
) -> str:
    try:
        args = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    query = str(args.get("query") or "").strip()
    if not query:
        return "Tool call was missing a query."

    if name == "search_uploads":
        return await search_uploads(settings, store, query)
    if name == "web_search":
        return await web_search(settings, query)
    return f"Unknown tool: {name}"
