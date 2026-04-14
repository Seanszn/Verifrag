"""Server-side orchestration pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from src.generation.ollama_backend import OllamaBackend
from src.indexing.bm25_index import BM25Index
from src.indexing.chroma_store import ChromaStore
from src.indexing.embedder import Embedder
from src.indexing.index_discovery import discover_index_artifacts
from src.retrieval.hybrid_retriever import HybridRetriever
from src.storage.database import Database
from src.verification.claim_decomposer import decompose_document
from src.verification.nli_verifier import AggregatedScore, NLIVerifier
from src.verification.verdict import classify_verification

CONVERSATION_CONTEXT_MESSAGE_LIMIT = 8


class QueryPipeline:
    """Owns the server-side query lifecycle."""

    def __init__(
        self,
        db: Database,
        llm: OllamaBackend | None = None,
        retriever: HybridRetriever | None = None,
        verifier: NLIVerifier | None = None,
        *,
        enable_verification: bool = True,
    ) -> None:
        self.db = db
        self.llm = llm or OllamaBackend()
        if retriever is None:
            self.retriever, self.retriever_status = _load_default_retriever()
        else:
            self.retriever = retriever
            self.retriever_status = "configured"
        self.verifier = verifier
        self.enable_verification = enable_verification

    def run(
        self,
        user_id: int,
        query: str,
        conversation_id: Optional[int] = None,
    ) -> dict[str, Any]:
        conversation = self._ensure_conversation(user_id, conversation_id, query)
        conversation_context = self._load_conversation_context(conversation["id"], user_id)
        user_message = self.db.add_message(conversation["id"], "user", query)

        assistant_text, pipeline_meta = self._generate_response(query, conversation_context)
        assistant_message = self.db.add_message(
            conversation["id"],
            "assistant",
            assistant_text,
            metadata_json=json.dumps(pipeline_meta),
        )
        conversation = self.db.get_conversation(conversation["id"], user_id) or conversation

        return {
            "conversation": conversation,
            "user_message": user_message,
            "assistant_message": assistant_message,
            "pipeline": pipeline_meta,
        }

    def _ensure_conversation(
        self,
        user_id: int,
        conversation_id: Optional[int],
        query: str,
    ) -> dict[str, Any]:
        if conversation_id is not None:
            conversation = self.db.get_conversation(conversation_id, user_id)
            if conversation is not None:
                return conversation
        return self.db.create_conversation(user_id, _default_title(query))

    def _load_conversation_context(
        self,
        conversation_id: int,
        user_id: int,
    ) -> list[dict[str, Any]]:
        messages = self.db.list_recent_messages(
            conversation_id,
            user_id,
            limit=CONVERSATION_CONTEXT_MESSAGE_LIMIT,
        )
        return [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in messages
            if message.get("content")
        ]

    def _generate_response(
        self,
        query: str,
        conversation_context: list[dict[str, Any]] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        retrieved_chunks = []
        retrieval_status = self.retriever_status
        retrieval_error = False

        if self.retriever is not None:
            try:
                retrieved_chunks = self.retriever.retrieve(query)
            except Exception as exc:  # pragma: no cover - defensive path for live index/runtime failures
                retrieval_status = f"error:{exc.__class__.__name__}"
                retrieval_error = True

        try:
            if retrieved_chunks:
                response = self.llm.generate_with_context(
                    query,
                    [_format_chunk_for_prompt(chunk) for chunk in retrieved_chunks],
                    conversation_history=conversation_context,
                )
                generation_mode = "rag"
            else:
                response = self.llm.generate_legal_answer(
                    query,
                    conversation_history=conversation_context,
                )
                generation_mode = "direct"
            backend_status = "ok"
        except Exception as exc:  # pragma: no cover - defensive path for live Ollama failures
            response = (
                "The backend could not reach the configured LLM provider. "
                "Check Ollama availability and server configuration."
            )
            backend_status = f"error:{exc.__class__.__name__}"
            generation_mode = "rag" if retrieved_chunks else "direct"

        meta = {
            "llm_provider": "ollama",
            "llm_backend_status": backend_status,
            "generation_mode": generation_mode,
            "retrieval_used": bool(retrieved_chunks),
            "retrieval_backend_status": retrieval_status,
            "retrieval_chunk_count": len(retrieved_chunks),
            "retrieved_chunks": [_serialize_chunk(chunk) for chunk in retrieved_chunks],
            "verification_enabled": self.enable_verification,
            "conversation_context_message_count": len(conversation_context or []),
        }

        if not self.enable_verification:
            meta["claim_count"] = 0
            meta["claims"] = []
            meta["verification_backend_status"] = "disabled:config"
            return response, meta

        raw_claims = decompose_document({"id": "assistant_response", "full_text": response})
        claims = [claim.to_dict() for claim in raw_claims]
        meta["claim_count"] = len(claims)
        meta["claims"] = claims
        meta["verification_backend_status"] = "skipped:no_retriever"

        if not raw_claims:
            meta["verification_backend_status"] = "skipped:no_claims"
            return response, meta

        if self.retriever is None:
            return response, meta

        if retrieval_error:
            meta["verification_backend_status"] = "skipped:retrieval_error"
            return response, meta

        if not retrieved_chunks:
            meta["verification_backend_status"] = "skipped:no_evidence"
            return response, meta

        verifier = self.verifier or NLIVerifier()
        try:
            verdicts = verifier.verify_claims_batch(raw_claims, retrieved_chunks)
        except Exception as exc:  # pragma: no cover - defensive path for live model/runtime failures
            meta["verification_backend_status"] = f"error:{exc.__class__.__name__}"
            meta["verification_error"] = {
                "type": exc.__class__.__name__,
                "message": str(exc),
            }
            return response, meta

        meta["claims"] = [
            _serialize_claim_with_verification(claim, verdict)
            for claim, verdict in zip(raw_claims, verdicts)
        ]
        meta["verification_backend_status"] = "ok"
        return response, meta


def _default_title(query: str) -> str:
    trimmed = " ".join(query.strip().split())
    if not trimmed:
        return "New conversation"
    if len(trimmed) <= 60:
        return trimmed
    return trimmed[:57].rstrip() + "..."


def _load_default_retriever() -> tuple[HybridRetriever | None, str]:
    artifacts = discover_index_artifacts()
    bm25_index = _load_bm25_index(artifacts.bm25_path)
    vector_store, embedder = _load_vector_store(
        artifacts.chroma_path,
        collection_name=artifacts.collection_name,
    )

    if bm25_index is None and vector_store is None:
        return None, "unavailable:no_indices"

    try:
        retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_index=bm25_index,
            embedder=embedder,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"error:{exc.__class__.__name__}"

    return retriever, "ok"


def _load_bm25_index(index_path: Path) -> BM25Index | None:
    if not index_path.exists():
        return None

    bm25_index = BM25Index(index_path=index_path)
    bm25_index.load()
    return bm25_index


def _load_vector_store(
    chroma_path: Path,
    *,
    collection_name: str | None = None,
) -> tuple[ChromaStore | None, Embedder | None]:
    if not chroma_path.exists():
        return None, None

    try:
        has_entries = any(chroma_path.iterdir())
    except OSError:
        has_entries = False

    if not has_entries:
        return None, None

    vector_store = ChromaStore(path=chroma_path, collection_name=collection_name)
    return vector_store, Embedder()


def _serialize_chunk(chunk) -> dict[str, Any]:
    payload = chunk.to_dict()
    payload["text_preview"] = chunk.text[:280]
    return payload


def _format_chunk_for_prompt(chunk) -> str:
    metadata = [
        f"Chunk ID: {chunk.id}",
        f"Document ID: {chunk.doc_id}",
        f"Document type: {chunk.doc_type}",
    ]
    if chunk.case_name:
        metadata.append(f"Case name: {chunk.case_name}")
    if chunk.court:
        metadata.append(f"Court: {chunk.court}")
    if chunk.court_level:
        metadata.append(f"Court level: {chunk.court_level}")
    if chunk.citation:
        metadata.append(f"Citation: {chunk.citation}")
    if chunk.date_decided:
        metadata.append(f"Date decided: {chunk.date_decided.isoformat()}")
    if chunk.title is not None:
        metadata.append(f"Title: {chunk.title}")
    if chunk.section:
        metadata.append(f"Section: {chunk.section}")
    if chunk.source_file:
        metadata.append(f"Source file: {chunk.source_file}")

    return f"{'; '.join(metadata)}\n{chunk.text}"


def _serialize_claim_with_verification(claim, verdict: AggregatedScore) -> dict[str, Any]:
    payload = claim.to_dict()
    classification = classify_verification(verdict)
    payload["verification"] = {
        "final_score": verdict.final_score,
        "verdict": classification.label,
        "verdict_explanation": classification.explanation,
        "is_contradicted": verdict.is_contradicted,
        "best_chunk_idx": verdict.best_chunk_idx,
        "support_ratio": verdict.support_ratio,
        "component_scores": verdict.component_scores,
        "best_chunk": _serialize_chunk(verdict.best_chunk) if verdict.best_chunk is not None else None,
    }
    return payload
