"""FastAPI dependencies."""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from src.storage.database import Database


database = Database()


def get_database() -> Database:
    return database


def require_user(
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
        )

    token = authorization.split(" ", 1)[1].strip()
    user = database.get_user_for_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        )
    user["token"] = token
    return user
