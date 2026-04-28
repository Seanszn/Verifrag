"""Query execution routes for the FastAPI backend."""

from __future__ import annotations

import logging
import time
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, Request

from src.api.dependencies import get_current_user, get_pipeline
from src.api.schemas import QueryRequest, QueryResponse
from src.pipeline import QueryPipeline


router = APIRouter(tags=["query"])
logger = logging.getLogger(__name__)


@router.post("/api/query", response_model=QueryResponse)
def submit_query(
    payload: QueryRequest,
    request: Request,
    current_user: dict[str, Any] = Depends(get_current_user),
    pipeline: QueryPipeline = Depends(get_pipeline),
    x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
) -> dict[str, Any]:
    request_id = (
        x_request_id
        or getattr(request.state, "request_id", None)
        or uuid4().hex
    ).strip()
    query_preview = " ".join(payload.query.split())[:120]
    started = time.perf_counter()
    logger.info(
        "query.request_start request_id=%s user_id=%s conversation_id=%s query=%r",
        request_id,
        current_user["id"],
        payload.conversation_id,
        query_preview,
    )
    try:
        result = pipeline.run(
            user_id=current_user["id"],
            query=payload.query,
            conversation_id=payload.conversation_id,
            request_id=request_id,
            include_uploaded_chunks=payload.include_uploaded_chunks,
        )
    except Exception:
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.exception(
            "query.request_error request_id=%s user_id=%s conversation_id=%s elapsed_ms=%.1f",
            request_id,
            current_user["id"],
            payload.conversation_id,
            elapsed_ms,
        )
        raise

    pipeline_meta = result.get("pipeline")
    if isinstance(pipeline_meta, dict):
        pipeline_meta.setdefault("request_id", request_id)

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "query.request_complete request_id=%s user_id=%s conversation_id=%s interaction_id=%s elapsed_ms=%.1f llm_status=%s retrieval_status=%s verification_status=%s timings_ms=%s",
        request_id,
        current_user["id"],
        result.get("conversation", {}).get("id"),
        result.get("interaction", {}).get("id"),
        elapsed_ms,
        pipeline_meta.get("llm_backend_status") if isinstance(pipeline_meta, dict) else None,
        pipeline_meta.get("retrieval_backend_status") if isinstance(pipeline_meta, dict) else None,
        pipeline_meta.get("verification_backend_status") if isinstance(pipeline_meta, dict) else None,
        pipeline_meta.get("timings_ms") if isinstance(pipeline_meta, dict) else None,
    )
    return result
