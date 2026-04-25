"""Import MongoDB documents and rebuild FAISS index."""

from __future__ import annotations

import argparse
import logging
from typing import Any

from app.config import load_config
from app.models.embedding_client import EmbeddingClient
from app.models.provider_factory import build_provider_client
from app.vector_store.metadata_store import MetadataItem, MetadataStore

LOGGER = logging.getLogger(__name__)
BATCH_SIZE = 10


def get_values_by_path(data: Any, path: str) -> list[Any]:
    """Extract values by dotted path with [] list wildcard."""
    parts = path.split(".")
    current: list[Any] = [data]
    for part in parts:
        next_values: list[Any] = []
        is_list_part = part.endswith("[]")
        key = part[:-2] if is_list_part else part
        for item in current:
            if not isinstance(item, dict) or key not in item:
                continue
            value = item[key]
            if is_list_part:
                if isinstance(value, list):
                    next_values.extend(value)
            else:
                next_values.append(value)
        current = next_values
    return current


def normalize_text_values(value: Any) -> list[str]:
    """Normalize scalar or list values to non-empty text list."""
    out: list[str] = []
    if isinstance(value, str):
        text = value.strip()
        if text:
            out.append(text)
        return out
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    out.append(text)
        return out
    if value is not None:
        text = str(value).strip()
        if text:
            out.append(text)
    return out


def build_text(document: dict[str, Any], text_fields: list[str]) -> str:
    """Build searchable text from configured fields."""
    # Schema is not fixed yet; choose fields in config.mongodb.text_fields.
    parts: list[str] = []
    for field in text_fields:
        values = get_values_by_path(document, field)
        for value in values:
            parts.extend(normalize_text_values(value))
    return "\n".join(parts)


def main() -> None:
    """Run full import and rebuild local vector index."""
    parser = argparse.ArgumentParser(description="Import MongoDB into FAISS")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    logging.basicConfig(
        level=logging.DEBUG if config.server.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    from pymongo import MongoClient

    from app.vector_store.faiss_store import FaissStore

    client = MongoClient(config.mongodb.uri)
    collection = client[config.mongodb.database][config.mongodb.collection]
    cursor = collection.find({})

    docs: list[dict[str, Any]] = list(cursor)
    if not docs:
        raise RuntimeError("No documents found from MongoDB collection")

    metadata_items: list[MetadataItem] = []
    batch_texts: list[str] = []
    store = None
    indexed_count = 0

    provider_client = build_provider_client(config.embedding)
    embedding_client = EmbeddingClient(provider_client, config.embedding.model)
    for doc in docs:
        text = build_text(doc, config.mongodb.text_fields)
        if not text:
            continue
        doc_id = str(doc.get(config.mongodb.id_field, ""))
        metadata_items.append(MetadataItem(doc_id=doc_id, text=text, source={}))
        batch_texts.append(text)
        if len(batch_texts) < BATCH_SIZE:
            continue
        vectors = embedding_client.embed_texts(batch_texts)
        if store is None:
            store = FaissStore.create(len(vectors[0]))
        store.add_vectors(vectors)
        indexed_count += len(batch_texts)
        LOGGER.info("Indexed batch size=%d total=%d", len(batch_texts), indexed_count)
        batch_texts = []

    if not metadata_items:
        raise RuntimeError("No non-empty text built from configured fields")

    if batch_texts:
        vectors = embedding_client.embed_texts(batch_texts)
        if store is None:
            store = FaissStore.create(len(vectors[0]))
        store.add_vectors(vectors)
        indexed_count += len(batch_texts)
        LOGGER.info("Indexed final batch size=%d total=%d", len(batch_texts), indexed_count)

    if store is None:
        raise RuntimeError("Vector index was not initialized")
    store.save(config.storage.index_path)
    MetadataStore.save(config.storage.metadata_path, metadata_items)
    LOGGER.info("Import complete. indexed=%d", len(metadata_items))


if __name__ == "__main__":
    main()
