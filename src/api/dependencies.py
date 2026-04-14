"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, Header, HTTPException, status

from src.pipeline import QueryPipeline
from src.storage.database import Database


database = Database()
database.initialize()

pipeline = QueryPipeline(db=database)
AUTHENTICATE_HEADERS = {"WWW-Authenticate": "Bearer"}


def get_db() -> Database:
    """Return the shared database instance."""
    return database


def get_pipeline() -> QueryPipeline:
    """Return the shared query pipeline instance."""
    return pipeline


def missing_bearer_token_exception() -> HTTPException:
    """Return the standardized missing-token response."""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing bearer token.",
        headers=AUTHENTICATE_HEADERS,
    )


def invalid_or_expired_token_exception() -> HTTPException:
    """Return the standardized invalid-token response."""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token.",
        headers=AUTHENTICATE_HEADERS,
    )


def get_bearer_token(authorization: str | None = Header(default=None)) -> str:
    """Extract the bearer token from the Authorization header."""
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
    """Resolve the current user from a bearer token."""
    user = db.get_user_for_token(token)
    if user is None:
        raise invalid_or_expired_token_exception()

    return user
