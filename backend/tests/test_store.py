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


def test_replace_swaps_the_complete_corpus():
    store = MemoryStore()
    store.add([Chunk("old.txt", "old", 0)], np.array([[1.0, 0.0]], dtype=np.float32))

    store.replace(
        [Chunk("new.txt", "new", 0)],
        np.array([[0.0, 1.0]], dtype=np.float32),
    )

    hits = store.search(np.array([0.0, 1.0]), top_k=1)
    assert [chunk.file_name for chunk, _score in hits] == ["new.txt"]
