"""Pydantic models for API requests and responses."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=256)


class LoginRequest(RegisterRequest):
    pass


class UserResponse(BaseModel):
    id: int
    username: str
    created_at: str


class AuthResponse(BaseModel):
    token: str
    user: UserResponse


class ConversationCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ConversationResponse(BaseModel):
    id: int
    user_id: int
    title: str
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    id: int
    conversation_id: int
    role: str
    content: str
    created_at: str
    metadata_json: Optional[str] = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    conversation_id: Optional[int] = None


class QueryResponse(BaseModel):
    conversation: ConversationResponse
    user_message: MessageResponse
    assistant_message: MessageResponse
    pipeline: dict[str, Any]
