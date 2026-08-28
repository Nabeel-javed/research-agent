"""Small worker pool so several files can embed at the same time.

Upload still waits until the jobs for *this* request finish. That way a
research call that starts immediately after upload does not race an empty store.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np

from app.config import get_settings
from app.llm.lm_studio import embed_texts
from app.store.memory import Chunk, MemoryStore


@dataclass
class EmbedJob:
    chunks: list[Chunk]
    done: asyncio.Future[np.ndarray]


class EmbedQueue:
    def __init__(self, store: MemoryStore, worker_count: int) -> None:
        self._store = store
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

    async def embed_and_store(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        loop = asyncio.get_running_loop()
        job = EmbedJob(chunks=chunks, done=loop.create_future())
        await self._queue.put(job)
        vectors = await job.done
        self._store.add(chunks, vectors)

    async def _worker(self) -> None:
        settings = get_settings()
        while True:
            job = await self._queue.get()
            try:
                texts = [chunk.text for chunk in job.chunks]
                vectors = await embed_texts(settings, texts)
                array = np.array(vectors, dtype=np.float32)
                if not job.done.done():
                    job.done.set_result(array)
            except Exception as exc:  # noqa: BLE001 — surface embed errors to upload
                if not job.done.done():
                    job.done.set_exception(exc)
            finally:
                self._queue.task_done()
