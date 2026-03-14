"""Tests for the memory subsystem (VectorStore)."""

from __future__ import annotations

import numpy as np
import pytest

from zeno.core.memory.vector_store import MemoryEntry, VectorStore

DIM = 8


@pytest.fixture()
def store() -> VectorStore:
    return VectorStore(embedding_dim=DIM, max_entries=10)


def _vec(*values: float) -> np.ndarray:
    """Build a float32 array padded / truncated to DIM."""
    arr = list(values) + [0.0] * (DIM - len(values))
    return np.array(arr[:DIM], dtype=np.float32)


class TestVectorStore:
    def test_add_and_size(self, store: VectorStore) -> None:
        store.add("hello", _vec(1, 0, 0, 0))
        assert store.size == 1

    def test_add_wrong_dim_raises(self, store: VectorStore) -> None:
        with pytest.raises(ValueError, match="Expected embedding"):
            store.add("bad", np.zeros(DIM + 1, dtype=np.float32))

    def test_search_returns_closest(self, store: VectorStore) -> None:
        store.add("alpha", _vec(1, 0, 0, 0))
        store.add("beta", _vec(0, 1, 0, 0))
        store.add("gamma", _vec(0, 0, 1, 0))

        results = store.search(_vec(1, 0, 0, 0), top_k=1)
        assert len(results) == 1
        entry, score = results[0]
        assert entry.text == "alpha"
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_search_empty_store(self, store: VectorStore) -> None:
        results = store.search(_vec(1, 0), top_k=3)
        assert results == []

    def test_search_zero_query(self, store: VectorStore) -> None:
        store.add("a", _vec(1, 0))
        results = store.search(np.zeros(DIM, dtype=np.float32), top_k=1)
        assert results == []

    def test_delete_existing(self, store: VectorStore) -> None:
        entry = store.add("to delete", _vec(1, 0))
        assert store.delete(entry.entry_id) is True
        assert store.size == 0

    def test_delete_nonexistent(self, store: VectorStore) -> None:
        assert store.delete("nonexistent-id") is False

    def test_clear(self, store: VectorStore) -> None:
        store.add("a", _vec(1))
        store.add("b", _vec(0, 1))
        store.clear()
        assert store.size == 0

    def test_eviction_on_overflow(self, store: VectorStore) -> None:
        for i in range(10):
            store.add(f"entry-{i}", _vec(float(i) / 10))
        # Store is at capacity (10); adding one more should evict oldest
        store.add("new", _vec(0.99))
        assert store.size == 10
        texts = [e.text for e, _ in store.search(_vec(0), top_k=10)]
        assert "entry-0" not in texts

    def test_metadata_preserved(self, store: VectorStore) -> None:
        meta = {"source": "test", "tag": "unit"}
        entry = store.add("text", _vec(1), metadata=meta)
        assert entry.metadata == meta
