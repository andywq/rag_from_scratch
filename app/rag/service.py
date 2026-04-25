"""RAG orchestration service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.models.embedding_client import EmbeddingClient
from app.models.llm_client import LLMClient
from app.vector_store.metadata_store import MetadataItem


class VectorStoreProtocol(Protocol):
    """Protocol for vector store search behavior."""

    def search(self, query_vector: list[float], top_k: int) -> list[object]:
        """Search top-k results."""


@dataclass(frozen=True)
class RAGHit:
    """One retrieval record returned to clients."""

    doc_id: str
    score: float
    text: str


@dataclass(frozen=True)
class RAGResponse:
    """Final response payload from RAG pipeline."""

    answer: str
    hits: list[RAGHit]


class RAGService:
    """Orchestrate retrieval and generation."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        llm_client: LLMClient,
        vector_store: VectorStoreProtocol,
        metadata_items: list[MetadataItem],
        top_k: int,
    ) -> None:
        self._embedding_client = embedding_client
        self._llm_client = llm_client
        self._vector_store = vector_store
        self._metadata_items = metadata_items
        self._top_k = top_k

    def answer_question(self, question: str) -> RAGResponse:
        """Answer question by retrieval-augmented generation."""
        vector = self._embedding_client.embed_texts([question])[0]
        raw_hits = self._vector_store.search(vector, self._top_k)
        hits: list[RAGHit] = []
        contexts: list[str] = []
        for hit in raw_hits:
            item = self._metadata_items[hit.index]
            hits.append(RAGHit(doc_id=item.doc_id, score=hit.score, text=item.text))
            contexts.append(item.text)
        answer = self._llm_client.summarize(question=question, contexts=contexts)
        return RAGResponse(answer=answer, hits=hits)
