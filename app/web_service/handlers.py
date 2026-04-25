"""HTTP handlers."""

from __future__ import annotations

from app.rag.service import RAGService
from app.web_service.schemas import QueryHit, QueryRequest, QueryResponse


def handle_query(request: QueryRequest, rag_service: RAGService) -> QueryResponse:
    """Handle query request and map service response to API schema."""
    result = rag_service.answer_question(request.query)
    return QueryResponse(
        answer=result.answer,
        hits=[
            QueryHit(doc_id=item.doc_id, score=item.score, text=item.text)
            for item in result.hits
        ],
    )
