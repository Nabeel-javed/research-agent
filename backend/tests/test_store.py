import numpy as np

from app.store.memory import Chunk, MemoryStore


def test_search_returns_closest_chunk():
    store = MemoryStore()
    store.add(
        [
            Chunk("a.txt", "alpha", 0),
            Chunk("b.txt", "beta", 0),
        ],
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    hits = store.search([0.9, 0.1], top_k=1)
    assert len(hits) == 1
    assert hits[0][0].file_name == "a.txt"


def test_clear_empties_corpus():
    store = MemoryStore()
    store.add([Chunk("a.txt", "alpha", 0)], np.array([[1.0, 0.0]], dtype=np.float32))
    store.clear()
    assert store.search(np.array([1.0, 0.0]), top_k=1) == []
