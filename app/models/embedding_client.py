"""Embedding model adapter."""

from __future__ import annotations

from typing import Protocol


class EmbeddingBackend(Protocol):
    """Protocol for embedding backend clients."""

    def create_embedding(self, model: str, texts: list[str]) -> list[list[float]]:
        """Create embedding vectors for text list."""


class EmbeddingClient:
    """Generate text embeddings via configured provider."""

    def __init__(self, client: EmbeddingBackend, model: str) -> None:
        self._client = client
        self._model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for given text list."""
        return self._client.create_embedding(self._model, texts)
