"""Provider-specific model clients."""

from __future__ import annotations

import json
from typing import Any
from urllib import request


class OpenAIClient:
    """Call OpenAI-compatible embedding and chat APIs."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    def create_embedding(self, model: str, texts: list[str]) -> list[list[float]]:
        """Create embeddings for input text list."""
        payload = {"model": model, "input": texts}
        body = self._post_json("/embeddings", payload)
        return [item["embedding"] for item in body["data"]]

    def create_chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
    ) -> str:
        """Create a chat completion and return answer text."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        body = self._post_json("/chat/completions", payload)
        return body["choices"][0]["message"]["content"]

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = request.Request(
            url=f"{self._base_url}{path}",
            data=data,
            method="POST",
            headers=headers,
        )
        with request.urlopen(req, timeout=60) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))


class OllamaClient:
    """Call Ollama native APIs for embedding and chat."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    def create_embedding(self, model: str, texts: list[str]) -> list[list[float]]:
        """Create embeddings in batch via Ollama embed endpoint."""
        payload = {"model": model, "input": texts}
        body = self._post_json("/api/embed", payload)
        return body["embeddings"]

    def create_chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
    ) -> str:
        """Create chat response via Ollama chat endpoint."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        body = self._post_json("/api/chat", payload)
        return body["message"]["content"]

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self._base_url}{path}",
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req, timeout=60) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
