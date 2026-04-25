"""Chat model adapter."""

from __future__ import annotations

from typing import Protocol


class ChatBackend(Protocol):
    """Protocol for chat backend clients."""

    def create_chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
    ) -> str:
        """Generate chat completion text."""


class LLMClient:
    """Generate answers with configured chat provider."""

    def __init__(self, client: ChatBackend, model: str) -> None:
        self._client = client
        self._model = model

    def summarize(self, question: str, contexts: list[str]) -> str:
        """Summarize answer from retrieved context."""
        joined_context = "\n\n".join(
            f"[Context {idx + 1}]\n{content}" for idx, content in enumerate(contexts)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise QA assistant. Use only the provided contexts. "
                    "If context is insufficient, say so clearly."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nContexts:\n{joined_context}",
            },
        ]
        return self._client.create_chat_completion(model=self._model, messages=messages)
