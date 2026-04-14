"""Conversation history routes for the FastAPI backend."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_current_user, get_db
from src.api.schemas import ConversationCreateRequest, ConversationSummary, MessageResponse
from src.storage.database import Database


router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationSummary])
def list_conversations(
    current_user: dict[str, Any] = Depends(get_current_user),
    db: Database = Depends(get_db),
) -> list[dict[str, Any]]:
    """Return the authenticated user's conversations."""
    return db.list_conversations(current_user["id"])


@router.post("", response_model=ConversationSummary, status_code=status.HTTP_201_CREATED)
def create_conversation(
    payload: ConversationCreateRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """Create a conversation owned by the authenticated user."""
    return db.create_conversation(current_user["id"], payload.title)


@router.get("/{conversation_id}/messages", response_model=list[MessageResponse])
def list_messages(
    conversation_id: int,
    current_user: dict[str, Any] = Depends(get_current_user),
    db: Database = Depends(get_db),
) -> list[dict[str, Any]]:
    """Return message history for a conversation owned by the current user."""
    conversation = db.get_conversation(conversation_id, current_user["id"])
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")

    return db.list_messages(conversation_id, current_user["id"])
