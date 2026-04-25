"""Tests for RAG service orchestration."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from app.rag.service import RAGService
from app.vector_store.metadata_store import MetadataItem


@dataclass(frozen=True)
class SearchHit:
    index: int
    score: float


class FakeEmbeddingClient:
    """Fake embedding client for tests."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


class FakeLLMClient:
    """Fake llm client for tests."""

    def summarize(self, question: str, contexts: list[str]) -> str:
        return f"Q:{question};C:{len(contexts)}"


class FakeVectorStore:
    """Fake vector store for tests."""

    def search(self, query_vector: list[float], top_k: int) -> list[SearchHit]:
        _ = query_vector
        _ = top_k
        return [SearchHit(index=1, score=0.91), SearchHit(index=0, score=0.88)]


class RAGServiceTestCase(unittest.TestCase):
    """Test RAG service output contract."""

    def test_answer_question(self) -> None:
        service = RAGService(
            embedding_client=FakeEmbeddingClient(),
            llm_client=FakeLLMClient(),
            vector_store=FakeVectorStore(),
            metadata_items=[
                MetadataItem(doc_id="doc-0", text="Text 0", source={}),
                MetadataItem(doc_id="doc-1", text="Text 1", source={}),
            ],
            top_k=2,
        )
        result = service.answer_question("hello")
        self.assertEqual(result.answer, "Q:hello;C:2")
        self.assertEqual(result.hits[0].doc_id, "doc-1")
        self.assertEqual(len(result.hits), 2)


if __name__ == "__main__":
    unittest.main()
