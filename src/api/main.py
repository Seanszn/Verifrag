"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request

from src.api import dependencies
from src.api.auth import router as auth_router
from src.api.conversations import router as conversations_router
from src.api.query import router as query_router
from src.api.uploads import router as uploads_router
from src.config import LOGGING


logger = logging.getLogger(__name__)


def _configured_log_level() -> int:
    level_name = LOGGING.level.strip().upper()
    configured_level = getattr(logging, level_name, logging.INFO)
    if not isinstance(configured_level, int):
        return logging.INFO
    return configured_level


def configure_app_logging() -> None:
    level = _configured_log_level()
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    logging.getLogger("src").setLevel(level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_app_logging()
    dependencies.database.initialize()
    app.state.database = dependencies.database
    app.state.pipeline = dependencies.pipeline
    preload_models = getattr(dependencies.pipeline, "preload_models", None)
    if callable(preload_models):
        app.state.model_preload_status = preload_models()
    else:
        app.state.model_preload_status = {"pipeline": "skipped:no_preload_hook"}
    logger.info(
        "api.startup verification_enabled=%s retrieval_status=%s llm_host=%s llm_model=%s preload_status=%s",
        getattr(dependencies.pipeline, "enable_verification", None),
        getattr(dependencies.pipeline, "retriever_status", None),
        getattr(getattr(dependencies.pipeline, "llm", None), "host", None),
        getattr(getattr(dependencies.pipeline, "llm", None), "model", None),
        app.state.model_preload_status,
    )
    try:
        yield
    finally:
        logger.info("api.shutdown")
        app.state.__dict__.pop("model_preload_status", None)
        app.state.__dict__.pop("pipeline", None)
        app.state.__dict__.pop("database", None)


app = FastAPI(title="VerifRAG API", lifespan=lifespan)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", "").strip() or uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()
    logger.info(
        "api.request_start request_id=%s method=%s path=%s client=%s",
        request_id,
        request.method,
        request.url.path,
        request.client.host if request.client is not None else None,
    )
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.exception(
            "api.request_error request_id=%s method=%s path=%s elapsed_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            elapsed_ms,
        )
        raise

    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-MS"] = f"{elapsed_ms:.1f}"
    log_fn = logger.warning if elapsed_ms >= LOGGING.slow_request_threshold_ms else logger.info
    log_fn(
        "api.request_complete request_id=%s method=%s path=%s status_code=%s elapsed_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


app.include_router(auth_router)
app.include_router(conversations_router)
app.include_router(query_router)
app.include_router(uploads_router)


@app.get("/health")
async def healthcheck() -> dict[str, object]:
    return {
        "status": "ok",
        "llm": dependencies.pipeline.llm.diagnostics(),
    }
