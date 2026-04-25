"""FastAPI application factory."""

from __future__ import annotations

from fastapi import Depends, FastAPI

from app.rag.service import RAGService
from app.web_service.handlers import handle_query
from app.web_service.schemas import QueryRequest, QueryResponse


def create_app(rag_service: RAGService) -> FastAPI:
    """Create FastAPI app with injected RAG service."""
    app = FastAPI(title="Minimal RAG Service")

    def get_rag_service() -> RAGService:
        return rag_service

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResponse)
    def query(
        payload: QueryRequest,
        service: RAGService = Depends(get_rag_service),
    ) -> QueryResponse:
        return handle_query(payload, service)

    return app
