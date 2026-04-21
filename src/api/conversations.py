"""Conversation history routes for the FastAPI backend."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_current_user, get_db
from src.api.schemas import (
    ConversationCreateRequest,
    ConversationSummary,
    InteractionDetailResponse,
    MessageResponse,
)
from src.storage.database import Database
from src.verification.claim_contract import normalize_claims_for_frontend


router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationSummary])
def list_conversations(
    current_user: dict[str, Any] = Depends(get_current_user),
    db: Database = Depends(get_db),
) -> list[dict[str, Any]]:
    return db.list_conversations(current_user["id"])


@router.post("", response_model=ConversationSummary, status_code=status.HTTP_201_CREATED)
def create_conversation(
    payload: ConversationCreateRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    return db.create_conversation(current_user["id"], payload.title)


@router.get("/{conversation_id}/messages", response_model=list[MessageResponse])
def list_messages(
    conversation_id: int,
    current_user: dict[str, Any] = Depends(get_current_user),
    db: Database = Depends(get_db),
) -> list[dict[str, Any]]:
    conversation = db.get_conversation(conversation_id, current_user["id"])
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    return db.list_messages(conversation_id, current_user["id"])


@router.get("/{conversation_id}/interactions", response_model=list[InteractionDetailResponse])
def list_interactions(
    conversation_id: int,
    current_user: dict[str, Any] = Depends(get_current_user),
    db: Database = Depends(get_db),
) -> list[dict[str, Any]]:
    conversation = db.get_conversation(conversation_id, current_user["id"])
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")

    interactions = db.list_interactions(conversation_id, current_user["id"])
    payloads: list[dict[str, Any]] = []
    for interaction in interactions:
        claims = [
            _decode_metadata_or_fallback(claim)
            for claim in db.list_verified_claims(interaction["id"], current_user["id"])
        ]
        claim_citation_links = [
            _decode_claim_citation_link(link)
            for link in db.list_claim_citation_links(interaction["id"], current_user["id"])
        ]
        normalized_claims, normalized_links = normalize_claims_for_frontend(
            claims,
            claim_citation_links=claim_citation_links,
        )
        payloads.append(
            {
                "interaction": interaction,
                "claims": normalized_claims,
                "citations": [
                    _decode_metadata_or_fallback(citation)
                    for citation in db.list_interaction_citations(interaction["id"], current_user["id"])
                ],
                "claim_citation_links": normalized_links,
                "contradictions": [
                    _decode_metadata_or_fallback(contradiction)
                    for contradiction in db.list_contradictions(interaction["id"], current_user["id"])
                ],
            }
        )
    return payloads


def _decode_metadata_or_fallback(row: dict[str, Any]) -> dict[str, Any]:
    metadata_json = row.get("metadata_json")
    if isinstance(metadata_json, str):
        try:
            payload = json.loads(metadata_json)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return payload

    fallback = dict(row)
    fallback.pop("metadata_json", None)

    span_json = fallback.get("span_json")
    if isinstance(span_json, str):
        try:
            fallback["span"] = json.loads(span_json)
        except json.JSONDecodeError:
            fallback["span"] = None
        fallback.pop("span_json", None)

    return fallback


def _decode_claim_citation_link(row: dict[str, Any]) -> dict[str, Any]:
    payload = _decode_metadata_or_fallback(row)
    payload.pop("citation_metadata_json", None)
    payload["claim_id"] = row.get("claim_id")
    payload["chunk_id"] = row.get("chunk_id")
    payload["doc_id"] = row.get("doc_id")
    payload["source_label"] = row.get("source_label")

    citation_payload = _decode_nested_metadata(
        row.get("citation_metadata_json"),
        fallback={
            "chunk_id": row.get("chunk_id"),
            "doc_id": row.get("doc_id"),
            "source_label": row.get("source_label"),
        },
    )
    payload["citation"] = citation_payload
    return payload


def _decode_nested_metadata(metadata_json: Any, *, fallback: dict[str, Any]) -> dict[str, Any]:
    if isinstance(metadata_json, str):
        try:
            payload = json.loads(metadata_json)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return payload
    return dict(fallback)
