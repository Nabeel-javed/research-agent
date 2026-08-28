"""FastAPI entrypoint.

The Vite app calls this process on port 8787. CORS has to allow the Vite origin
because the browser treats 5173 and 8787 as two different sites.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, RedirectResponse

from app.config import get_settings
from app.ingest.queue import EmbedQueue
from app.routes.research import router as research_router
from app.routes.sources import router as sources_router
from app.store.memory import MemoryStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    store = MemoryStore()
    queue = EmbedQueue(store=store, worker_count=settings.embed_workers)
    await queue.start()
    app.state.settings = settings
    app.state.store = store
    app.state.embed_queue = queue
    try:
        yield
    finally:
        await queue.stop()


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(title="Research agent backend", lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(sources_router)
    application.include_router(research_router)

    # Opening :8787 in a browser used to show a JSON 404. Send people to the UI.
    @application.get("/")
    def root() -> RedirectResponse:
        return RedirectResponse("http://localhost:5173/")

    @application.exception_handler(HTTPException)
    async def http_error(_request, exc: HTTPException) -> PlainTextResponse:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return PlainTextResponse(detail, status_code=exc.status_code)

    @application.exception_handler(RequestValidationError)
    async def validation_error(_request, exc: RequestValidationError) -> PlainTextResponse:
        return PlainTextResponse(str(exc.errors()), status_code=422)

    return application


app = create_app()
