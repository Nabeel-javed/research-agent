"""Create bounded, overlapping chunks for retrieval."""

from __future__ import annotations

from app.store.memory import Chunk


def chunk_text(file_name: str, text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    cleaned = text.replace("\r\n", "\n").strip()
    if not cleaned:
        return []
    pieces = _window(cleaned, chunk_size, overlap)
    return [Chunk(file_name=file_name, text=piece, index=i) for i, piece in enumerate(pieces)]


def _window(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size")
    if len(text) <= chunk_size:
        return [text]

    out: list[str] = []
    start = 0
    while start < len(text):
        maximum_end = min(start + chunk_size, len(text))
        end = _preferred_end(text, start, maximum_end, chunk_size)
        out.append(text[start:end].strip())
        if end >= len(text):
            break

        next_start = max(start + 1, end - overlap)
        while next_start > start and not text[next_start - 1].isspace():
            next_start -= 1
        if next_start <= start:
            next_start = max(start + 1, end - overlap)
        start = next_start
    return [item for item in out if item]


def _preferred_end(text: str, start: int, maximum_end: int, chunk_size: int) -> int:
    if maximum_end >= len(text):
        return len(text)

    minimum_end = start + max(chunk_size // 2, 1)
    boundaries = (("\n\n", 2), ("\n", 1), (". ", 1), (" ", 1))
    for marker, offset in boundaries:
        boundary = text.rfind(marker, minimum_end, maximum_end)
        if boundary >= minimum_end:
            return boundary + offset
    return maximum_end
