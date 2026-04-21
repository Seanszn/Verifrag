"""Pydantic schemas for the FastAPI backend."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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


class ClaimSpanResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    doc_id: str | None = None
    para_id: int | None = None
    sent_id: int | None = None
    start_char: int
    end_char: int
    text: str | None = None


class ClaimEvidenceResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    relationship: Literal["supporting", "contradicting"]
    score: float | None = None
    chunk_id: str | None = None
    doc_id: str | None = None
    source_label: str | None = None
    citation: dict[str, Any] = Field(default_factory=dict)


class ClaimAnnotationResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    support_level: Literal["supported", "possibly_supported", "unsupported"]
    explanation: str
    response_span: ClaimSpanResponse | None = None
    evidence: list[ClaimEvidenceResponse] = Field(default_factory=list)


class ClaimResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    claim_id: str
    text: str
    claim_type: str | None = None
    source: str | None = None
    certainty: str | None = None
    doc_section: str | None = None
    span: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    linked_citations: list[dict[str, Any]] = Field(default_factory=list)
    annotation: ClaimAnnotationResponse | None = None


class PipelineResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    claim_count: int = 0
    claims: list[ClaimResponse] = Field(default_factory=list)


class InteractionDetailResponse(BaseModel):
    interaction: InteractionResponse
    claims: list[ClaimResponse]
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
    pipeline: PipelineResponse
