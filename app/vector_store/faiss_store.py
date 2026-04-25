"""FAISS vector storage wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass(frozen=True)
class SearchHit:
    """One search hit from vector index."""

    index: int
    score: float


class FaissStore:
    """Read, write and search FAISS index."""

    def __init__(self, index: faiss.Index) -> None:
        self._index = index

    @classmethod
    def create(cls, dimension: int) -> "FaissStore":
        """Create an empty index with known vector dimension."""
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        index = faiss.IndexFlatIP(dimension)
        return cls(index)

    @classmethod
    def build(cls, vectors: list[list[float]]) -> "FaissStore":
        """Build a new index from vectors."""
        if not vectors:
            raise ValueError("vectors must not be empty")
        matrix = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return cls(index)

    @classmethod
    def load(cls, index_path: str) -> "FaissStore":
        """Load index from disk."""
        index = faiss.read_index(index_path)
        return cls(index)

    def save(self, index_path: str) -> None:
        """Persist index to disk."""
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, index_path)

    def add_vectors(self, vectors: list[list[float]]) -> None:
        """Add vectors to existing index."""
        if not vectors:
            return
        matrix = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(matrix)
        self._index.add(matrix)

    def search(self, query_vector: list[float], top_k: int) -> list[SearchHit]:
        """Search top-k nearest vectors."""
        query = np.asarray([query_vector], dtype="float32")
        faiss.normalize_L2(query)
        scores, indexes = self._index.search(query, top_k)
        hits: list[SearchHit] = []
        for idx, score in zip(indexes[0], scores[0], strict=True):
            if idx < 0:
                continue
            hits.append(SearchHit(index=int(idx), score=float(score)))
        return hits
