"""POST /api/research — matches src/api.ts streamResearch().

The browser reads response.body as raw UTF-8. Do not wrap this in SSE
(data: ...) frames; api.ts would print those frames as text.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.loop import run_research

router = APIRouter()


class ResearchBody(BaseModel):
    request: str = Field(min_length=1)


@router.post("/api/research")
async def research(request: Request, body: ResearchBody) -> StreamingResponse:
    question = body.request.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Research request is empty.")

    settings = request.app.state.settings
    store = request.app.state.store

    async def chunks():
        try:
            async for piece in run_research(settings, store, question):
                if await request.is_disconnected():
                    break
                if piece:
                    yield piece.encode("utf-8")
        except Exception as exc:  # noqa: BLE001
            yield f"\n\nResearch failed: {exc}".encode("utf-8")

    return StreamingResponse(chunks(), media_type="text/plain; charset=utf-8")
