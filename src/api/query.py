"""Query execution routes for the FastAPI backend."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from src.api.dependencies import get_current_user, get_pipeline
from src.api.schemas import QueryRequest, QueryResponse
from src.pipeline import QueryPipeline


router = APIRouter(tags=["query"])


@router.post("/api/query", response_model=QueryResponse)
def submit_query(
    payload: QueryRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    pipeline: QueryPipeline = Depends(get_pipeline),
) -> dict[str, Any]:
    return pipeline.run(
        user_id=current_user["id"],
        query=payload.query,
        conversation_id=payload.conversation_id,
    )
