"""Authentication routes for the FastAPI backend."""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status

from src.api.dependencies import (
    AUTHENTICATE_HEADERS,
    get_bearer_token,
    get_db,
    invalid_or_expired_token_exception,
)
from src.api.schemas import AuthResponse, LoginRequest, RegisterRequest
from src.auth.local_auth import hash_password, verify_password
from src.storage.database import Database


router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(
    payload: RegisterRequest,
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    if db.get_user_by_username(payload.username) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists.")

    try:
        user = db.create_user(payload.username, hash_password(payload.password))
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists.") from exc

    token = db.create_session(user["id"])
    return {"token": token, "user": user}


@router.post("/login", response_model=AuthResponse)
def login(
    payload: LoginRequest,
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    user_row = db.get_user_by_username(payload.username)
    if user_row is None or not verify_password(payload.password, user_row["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers=AUTHENTICATE_HEADERS,
        )

    user = {
        "id": user_row["id"],
        "username": user_row["username"],
        "email": user_row["email"],
        "created_at": user_row["created_at"],
    }
    token = db.create_session(user["id"])
    return {"token": token, "user": user}


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    token: str = Depends(get_bearer_token),
    db: Database = Depends(get_db),
) -> Response:
    if db.get_user_for_token(token) is None:
        raise invalid_or_expired_token_exception()

    db.delete_session(token)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
