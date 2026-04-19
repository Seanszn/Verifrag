"""Pydantic schemas for the FastAPI backend."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class UserResponse(BaseModel):
    id: int
    username: str
    email: str | None = None
    created_at: str


class AuthResponse(BaseModel):
    token: str
    user: UserResponse


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=256)


class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=256)


class ConversationCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ConversationSummary(BaseModel):
    id: int
    user_id: int
    title: str
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    id: int
    conversation_id: int
    interaction_id: int | None = None
    role: str
    content: str
    created_at: str
    metadata_json: str | None = None


class InteractionResponse(BaseModel):
    id: int
    conversation_id: int
    query: str
    response: str | None = None
    created_at: str


class InteractionDetailResponse(BaseModel):
    interaction: InteractionResponse
    claims: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    claim_citation_links: list[dict[str, Any]]
    contradictions: list[dict[str, Any]]


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    conversation_id: int | None = None


class QueryResponse(BaseModel):
    conversation: ConversationSummary
    interaction: InteractionResponse
    user_message: MessageResponse
    assistant_message: MessageResponse
    pipeline: dict[str, Any]
