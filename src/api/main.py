"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api import dependencies
from src.api.auth import router as auth_router
from src.api.conversations import router as conversations_router
from src.api.query import router as query_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    dependencies.database.initialize()
    app.state.database = dependencies.database
    app.state.pipeline = dependencies.pipeline
    try:
        yield
    finally:
        app.state.__dict__.pop("pipeline", None)
        app.state.__dict__.pop("database", None)


app = FastAPI(title="VerifRAG API", lifespan=lifespan)
app.include_router(auth_router)
app.include_router(conversations_router)
app.include_router(query_router)


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
