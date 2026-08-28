"""Process-local vector store for the active source collection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Chunk:
    file_name: str
    text: str
    index: int


@dataclass
class MemoryStore:
    chunks: list[Chunk] = field(default_factory=list)
    # One row per chunk. Empty until the first successful embed.
    vectors: np.ndarray | None = None

    def clear(self) -> None:
        self.chunks.clear()
        self.vectors = None

    def add(self, new_chunks: list[Chunk], new_vectors: np.ndarray) -> None:
        if len(new_chunks) != len(new_vectors):
            raise ValueError("Each chunk needs exactly one embedding.")
        self.chunks.extend(new_chunks)
        if self.vectors is None:
            self.vectors = new_vectors.astype(np.float32, copy=True)
        else:
            self.vectors = np.vstack([self.vectors, new_vectors.astype(np.float32)])

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        if self.vectors is None or not self.chunks:
            return []

        query = np.asarray(query_vector, dtype=np.float32)
        query_norm = np.linalg.norm(query) or 1.0
        matrix = self.vectors
        doc_norms = np.linalg.norm(matrix, axis=1)
        doc_norms[doc_norms == 0] = 1.0
        scores = (matrix @ query) / (doc_norms * query_norm)
        k = min(top_k, len(self.chunks))
        best = np.argpartition(-scores, kth=k - 1)[:k]
        best = best[np.argsort(-scores[best])]
        return [(self.chunks[i], float(scores[i])) for i in best]
