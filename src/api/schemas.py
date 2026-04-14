"""Shared request and response models for the FastAPI surface."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


UsernameStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=3, max_length=64),
]
PasswordStr = Annotated[
    str,
    StringConstraints(min_length=8, max_length=256),
]
ConversationTitleStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
]
QueryTextStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=8000),
]


class UserResponse(BaseModel):
    """Public user payload returned by auth endpoints."""

    id: int
    username: str
    email: str | None = None
    created_at: str


class RegisterRequest(BaseModel):
    """Register a new local account."""

    model_config = ConfigDict(extra="forbid")

    username: UsernameStr
    password: PasswordStr


class LoginRequest(BaseModel):
    """Authenticate an existing local account."""

    model_config = ConfigDict(extra="forbid")

    username: UsernameStr
    password: PasswordStr


class AuthResponse(BaseModel):
    """Session token plus the authenticated user."""

    token: str
    user: UserResponse


class ConversationCreateRequest(BaseModel):
    """Create a conversation directly via the API."""

    model_config = ConfigDict(extra="forbid")

    title: ConversationTitleStr


class ConversationSummary(BaseModel):
    """Conversation metadata for sidebar/history views."""

    id: int
    user_id: int
    title: str
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    """One persisted conversation message."""

    id: int
    conversation_id: int
    interaction_id: int | None = None
    role: str
    content: str
    created_at: str
    metadata_json: str | None = None


class InteractionResponse(BaseModel):
    """One normalized query/response interaction."""

    id: int
    conversation_id: int
    query: str
    response: str | None = None
    created_at: str


class QueryRequest(BaseModel):
    """Submit a legal question to the backend pipeline."""

    model_config = ConfigDict(extra="forbid")

    query: QueryTextStr
    conversation_id: int | None = None


class QueryResponse(BaseModel):
    """Combined response returned by the query endpoint."""

    conversation: ConversationSummary
    interaction: InteractionResponse
    user_message: MessageResponse
    assistant_message: MessageResponse
    pipeline: dict[str, Any] = Field(default_factory=dict)
