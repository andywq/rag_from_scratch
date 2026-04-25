"""Configuration models and loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MongoConfig:
    """MongoDB source configuration."""

    uri: str
    database: str
    collection: str
    text_fields: list[str]
    id_field: str = "_id"


@dataclass(frozen=True)
class ProviderConfig:
    """Model provider configuration."""

    provider: str
    base_url: str
    model: str
    api_key: str = ""


@dataclass(frozen=True)
class RetrievalConfig:
    """Vector retrieval configuration."""

    top_k: int = 3


@dataclass(frozen=True)
class StorageConfig:
    """Local storage paths for vector data."""

    index_path: str
    metadata_path: str


@dataclass(frozen=True)
class ServerConfig:
    """HTTP server runtime configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    debug: bool = False


@dataclass(frozen=True)
class AppConfig:
    """Root application configuration."""

    mongodb: MongoConfig
    chat: ProviderConfig
    embedding: ProviderConfig
    retrieval: RetrievalConfig
    storage: StorageConfig
    server: ServerConfig


def load_config(path: str) -> AppConfig:
    """Load configuration from a JSON file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    mongodb = MongoConfig(**raw["mongodb"])
    providers = raw["providers"]
    chat = ProviderConfig(**providers["chat"])
    embedding = ProviderConfig(**providers["embedding"])
    retrieval = RetrievalConfig(**raw.get("retrieval", {}))
    storage = StorageConfig(**raw["storage"])
    server = ServerConfig(**raw.get("server", {}))
    return AppConfig(
        mongodb=mongodb,
        chat=chat,
        embedding=embedding,
        retrieval=retrieval,
        storage=storage,
        server=server,
    )
