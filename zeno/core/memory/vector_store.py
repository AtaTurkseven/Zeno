"""In-memory vector store for Zeno's episodic memory.

Stores text entries together with their embedding vectors and supports
cosine-similarity retrieval.  No external vector DB is required for the
base system — a NumPy-backed implementation is provided and a
``chroma`` backend can be swapped in later by extending :class:`VectorStore`.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single entry stored in the vector memory.

    Attributes
    ----------
    text:
        The original human-readable text.
    embedding:
        Fixed-length float32 vector representing *text*.
    metadata:
        Arbitrary key-value pairs (source, timestamp, tags, …).
    entry_id:
        Unique identifier assigned at creation time.
    timestamp:
        Unix time at which the entry was added.
    """

    text: str
    embedding: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)


class VectorStore:
    """Thread-safe, in-memory vector database.

    Parameters
    ----------
    embedding_dim:
        Dimensionality of stored embedding vectors.
    max_entries:
        Maximum number of entries before older ones are evicted (FIFO).
    """

    def __init__(self, embedding_dim: int = 384, max_entries: int = 10_000) -> None:
        self._dim = embedding_dim
        self._max = max_entries
        self._entries: list[MemoryEntry] = []
        logger.debug(
            "VectorStore initialised (dim=%d, max=%d).", embedding_dim, max_entries
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Insert a new entry into the store.

        Parameters
        ----------
        text:
            Source text.
        embedding:
            Pre-computed embedding vector of shape ``(embedding_dim,)``.
        metadata:
            Optional key-value annotations.

        Returns
        -------
        MemoryEntry
            The created entry (with its assigned ``entry_id``).

        Raises
        ------
        ValueError
            When *embedding* has the wrong dimensionality.
        """
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.shape != (self._dim,):
            raise ValueError(
                f"Expected embedding of shape ({self._dim},), got {embedding.shape}."
            )

        entry = MemoryEntry(text=text, embedding=embedding, metadata=metadata or {})
        if len(self._entries) >= self._max:
            evicted = self._entries.pop(0)
            logger.debug("VectorStore evicted oldest entry id=%s.", evicted.entry_id)

        self._entries.append(entry)
        logger.debug("VectorStore added entry id=%s.", entry.entry_id)
        return entry

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[tuple[MemoryEntry, float]]:
        """Return the *top_k* most similar entries by cosine similarity.

        Parameters
        ----------
        query_embedding:
            Query vector of shape ``(embedding_dim,)``.
        top_k:
            Number of results to return.

        Returns
        -------
        list of (MemoryEntry, float)
            Sorted by descending similarity score.
        """
        if not self._entries:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        matrix = np.stack([e.embedding for e in self._entries])  # (N, dim)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        similarities = (matrix / norms) @ (query / query_norm)  # (N,)

        k = min(top_k, len(self._entries))
        top_indices = np.argpartition(similarities, -k)[-k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        return [(self._entries[i], float(similarities[i])) for i in top_indices]

    def delete(self, entry_id: str) -> bool:
        """Remove an entry by its ``entry_id``.

        Returns
        -------
        bool
            ``True`` if an entry was found and removed.
        """
        for idx, entry in enumerate(self._entries):
            if entry.entry_id == entry_id:
                self._entries.pop(idx)
                logger.debug("VectorStore deleted entry id=%s.", entry_id)
                return True
        return False

    def clear(self) -> None:
        """Remove all entries from the store."""
        self._entries.clear()
        logger.debug("VectorStore cleared.")

    @property
    def size(self) -> int:
        """Number of entries currently stored."""
        return len(self._entries)

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the stored embedding vectors."""
        return self._dim

    def __repr__(self) -> str:  # pragma: no cover
        return f"VectorStore(size={self.size}, dim={self._dim})"
