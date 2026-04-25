"""HTTP request and response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for query endpoint."""

    query: str = Field(min_length=1, description="User query string")


class QueryHit(BaseModel):
    """One hit item in query response."""

    doc_id: str
    score: float
    text: str


class QueryResponse(BaseModel):
    """Response body for query endpoint."""

    answer: str
    hits: list[QueryHit]
