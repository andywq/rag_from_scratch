"""Application entrypoint."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from app.config import AppConfig, load_config
from app.models.embedding_client import EmbeddingClient
from app.models.llm_client import LLMClient
from app.models.provider_factory import build_provider_client
from app.rag.service import RAGService
from app.vector_store.faiss_store import FaissStore
from app.vector_store.metadata_store import MetadataStore
from app.web_service.app import create_app


def build_rag_service(config: AppConfig) -> RAGService:
    """Build RAG service from app configuration."""
    chat_provider_client = build_provider_client(config.chat)
    embedding_provider_client = build_provider_client(config.embedding)
    embedding_client = EmbeddingClient(embedding_provider_client, config.embedding.model)
    llm_client = LLMClient(chat_provider_client, config.chat.model)
    vector_store = FaissStore.load(config.storage.index_path)
    metadata_items = MetadataStore.load(config.storage.metadata_path)
    return RAGService(
        embedding_client=embedding_client,
        llm_client=llm_client,
        vector_store=vector_store,
        metadata_items=metadata_items,
        top_k=config.retrieval.top_k,
    )


def main() -> None:
    """Start the FastAPI server."""
    parser = argparse.ArgumentParser(description="Run minimal RAG service")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, config.server.log_level.upper(),
                      logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    rag_service = build_rag_service(config)
    app = create_app(rag_service)
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level,
    )


if __name__ == "__main__":
    main()
