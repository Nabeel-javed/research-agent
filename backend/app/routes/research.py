"""POST /api/research — matches src/api.ts streamResearch().

The browser reads response.body as raw UTF-8. Do not wrap this in SSE
(data: ...) frames; api.ts would print those frames as text.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.loop import (
    RESEARCH_INTERRUPTED,
    RESEARCH_UNAVAILABLE,
    ResearchInterruptedError,
    ResearchUnavailableError,
    run_research,
)

router = APIRouter()
logger = logging.getLogger(__name__)


class ResearchBody(BaseModel):
    request: str = Field(min_length=1)


@router.post("/api/research")
async def research(request: Request, body: ResearchBody) -> StreamingResponse:
    question = body.request.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Research request is empty.")

    settings = request.app.state.settings
    store = request.app.state.store
    stream = run_research(settings, store, question)

    try:
        first_piece = await anext(stream)
    except StopAsyncIteration as exc:
        raise HTTPException(status_code=502, detail=RESEARCH_UNAVAILABLE) from exc
    except ResearchUnavailableError as exc:
        raise HTTPException(status_code=502, detail=RESEARCH_UNAVAILABLE) from exc
    except Exception as exc:
        logger.exception("Research failed before streaming began")
        raise HTTPException(status_code=500, detail=RESEARCH_UNAVAILABLE) from exc

    async def chunks():
        try:
            if first_piece:
                yield first_piece.encode("utf-8")
            async for piece in stream:
                if await request.is_disconnected():
                    break
                if piece:
                    yield piece.encode("utf-8")
        except ResearchInterruptedError:
            yield f"\n\n> {RESEARCH_INTERRUPTED}".encode()
        except Exception:
            logger.exception("Research failed after streaming began")
            yield f"\n\n> {RESEARCH_INTERRUPTED}".encode()
        finally:
            await stream.aclose()

    return StreamingResponse(chunks(), media_type="text/plain; charset=utf-8")
