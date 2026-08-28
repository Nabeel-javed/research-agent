"""Source upload endpoint matching the frontend response contract.

Embedding completes before the response is returned, preventing a subsequent
research request from racing an unfinished ingestion job.
"""

from __future__ import annotations

import asyncio

import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.ingest.chunking import chunk_text
from app.ingest.extract import ExtractError, extract_text, is_supported

router = APIRouter()


@router.post("/api/sources")
async def upload_sources(
    request: Request,
    files: list[UploadFile] = File(...),
) -> dict:
    settings = request.app.state.settings
    store = request.app.state.store
    queue = request.app.state.embed_queue

    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")
    if len(files) > settings.max_files_per_upload:
        raise HTTPException(
            status_code=400,
            detail=f"Upload at most {settings.max_files_per_upload} files at a time.",
        )

    # Extract everything first. A bad PDF must not wipe a previous good batch.
    pending: list[tuple[dict, list]] = []

    for upload in files:
        raw = await upload.read()
        if len(raw) > settings.max_file_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"{upload.filename} is larger than {settings.max_file_bytes} bytes.",
            )
        if not is_supported(upload.filename or "", upload.content_type):
            raise HTTPException(
                status_code=400,
                detail=f"{upload.filename} must be .txt, .md, or .pdf.",
            )
        try:
            text = extract_text(upload.filename or "upload", raw, upload.content_type)
        except ExtractError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        chunks = chunk_text(
            upload.filename or "upload.txt",
            text,
            settings.chunk_size,
            settings.chunk_overlap,
        )
        if not chunks:
            raise HTTPException(status_code=400, detail=f"{upload.filename} is empty.")

        pending.append(
            (
                {
                    "name": upload.filename or "upload.txt",
                    "size": len(raw),
                    "type": upload.content_type or "text/plain",
                },
                chunks,
            )
        )

    try:
        vectors_by_file = await asyncio.gather(
            *(queue.embed(chunks) for _, chunks in pending)
        )
        all_chunks = [chunk for _, chunks in pending for chunk in chunks]
        all_vectors = np.vstack(vectors_by_file)
        store.replace(all_chunks, all_vectors)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Source processing failed. Existing sources were preserved.",
        ) from exc

    return {"uploaded": [meta for meta, _ in pending]}
