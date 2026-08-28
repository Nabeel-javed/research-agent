"""Small worker pool so several files can embed at the same time.

Upload still waits until the jobs for *this* request finish. That way a
research call that starts immediately after upload does not race an empty store.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np

from app.config import get_settings
from app.embeddings.fastembed import embed_documents
from app.store.memory import Chunk


@dataclass
class EmbedJob:
    chunks: list[Chunk]
    done: asyncio.Future[np.ndarray]


class EmbedQueue:
    def __init__(self, worker_count: int) -> None:
        self._worker_count = worker_count
        self._queue: asyncio.Queue[EmbedJob] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []

    async def start(self) -> None:
        for _ in range(self._worker_count):
            self._workers.append(asyncio.create_task(self._worker()))

    async def stop(self) -> None:
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def embed(self, chunks: list[Chunk]) -> np.ndarray:
        if not chunks:
            return np.empty((0, 0), dtype=np.float32)
        loop = asyncio.get_running_loop()
        job = EmbedJob(chunks=chunks, done=loop.create_future())
        await self._queue.put(job)
        return await job.done

    async def _worker(self) -> None:
        settings = get_settings()
        while True:
            job = await self._queue.get()
            try:
                texts = [chunk.text for chunk in job.chunks]
                array = await embed_documents(settings, texts)
                if not job.done.done():
                    job.done.set_result(array)
            except Exception as exc:  # noqa: BLE001 — surface embed errors to upload
                if not job.done.done():
                    job.done.set_exception(exc)
            finally:
                self._queue.task_done()
