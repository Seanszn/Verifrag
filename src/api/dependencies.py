"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, Header, HTTPException, status

from src.config import VERIFICATION
from src.pipeline import QueryPipeline
from src.storage.database import Database


database = Database()
database.initialize()

pipeline = QueryPipeline(db=database, enable_verification=VERIFICATION.enabled)
AUTHENTICATE_HEADERS = {"WWW-Authenticate": "Bearer"}


def get_db() -> Database:
    return database


def get_pipeline() -> QueryPipeline:
    return pipeline


def missing_bearer_token_exception() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing bearer token.",
        headers=AUTHENTICATE_HEADERS,
    )


def invalid_or_expired_token_exception() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token.",
        headers=AUTHENTICATE_HEADERS,
    )


def get_bearer_token(authorization: str | None = Header(default=None)) -> str:
    if not authorization:
        raise missing_bearer_token_exception()

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise missing_bearer_token_exception()

    return token.strip()


def get_current_user(
    token: str = Depends(get_bearer_token),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    user = db.get_user_for_token(token)
    if user is None:
        raise invalid_or_expired_token_exception()
    return user
