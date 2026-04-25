"""Factory for building provider clients."""

from __future__ import annotations

from app.config import ProviderConfig
from app.models.provider_clients import OllamaClient, OpenAIClient


def build_provider_client(provider: ProviderConfig) -> object:
    """Build provider client instance from config."""
    if provider.provider == "openai":
        return OpenAIClient(base_url=provider.base_url, api_key=provider.api_key)
    if provider.provider == "ollama":
        return OllamaClient(base_url=provider.base_url)
    raise ValueError(f"Unsupported provider: {provider.provider}")
