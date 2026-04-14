"""Claim-level trace for a previously captured live Ollama response."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Sequence

import pytest

from scripts.nli_test_utils import HeuristicNLIVerifier
from src.ingestion.document import LegalChunk
from src.pipeline import QueryPipeline
from src.storage.database import Database


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_TRACE_PATH = PROJECT_ROOT / "artifacts" / "test_reports" / "ollama_legal_question_trace.json"
CLAIM_TRACE_PATH = PROJECT_ROOT / "artifacts" / "test_reports" / "ollama_legal_question_claims_trace.json"


class _TraceResponseLLM:
    """Returns the captured Ollama response while preserving pipeline call shape."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.context_calls: list[dict] = []
        self.direct_queries: list[str] = []

    def generate_with_context(
        self,
        query: str,
        context: Sequence[str],
        max_tokens: int | None = None,
    ) -> str:
        self.context_calls.append(
            {
                "query": query,
                "context": list(context),
                "max_tokens": max_tokens,
            }
        )
        return self.response

    def generate_legal_answer(self, query: str) -> str:
        self.direct_queries.append(query)
        return self.response


class _TraceContextRetriever:
    """Adapts captured prompt context into the retriever interface."""

    def __init__(self, chunks: list[LegalChunk]) -> None:
        self.chunks = chunks
        self.queries: list[dict] = []

    def retrieve(self, query: str, k: int = 10) -> list[LegalChunk]:
        self.queries.append({"query": query, "k": k})
        return self.chunks[:k]


def test_ollama_trace_output_decomposes_to_claim_level(tmp_path: Path):
    if not SOURCE_TRACE_PATH.exists():
        pytest.skip(f"Run the live Ollama trace test first: {SOURCE_TRACE_PATH}")

    source_trace = json.loads(SOURCE_TRACE_PATH.read_text(encoding="utf-8"))
    query = source_trace["input"]["query"]
    response = source_trace["output"]["response"]
    chunks = _context_chunks_from_trace(source_trace)

    db = Database(tmp_path / "ollama_claim_trace.db")
    db.initialize()
    user = db.create_user("ollama_claim_trace", "not-a-real-hash")

    llm = _TraceResponseLLM(response)
    retriever = _TraceContextRetriever(chunks)
    pipeline = QueryPipeline(
        db=db,
        llm=llm,
        retriever=retriever,
        verifier=HeuristicNLIVerifier(),
    )

    result = pipeline.run(user_id=user["id"], query=query)
    pipeline_meta = result["pipeline"]

    report = {
        "source_trace_path": str(SOURCE_TRACE_PATH),
        "source_document": source_trace["document"],
        "input": source_trace["input"],
        "captured_ollama_output": response,
        "pipeline_components": [
            "src.pipeline.QueryPipeline.run",
            "src.pipeline.QueryPipeline._generate_response",
            "src.verification.claim_decomposer.decompose_document",
            "scripts.nli_test_utils.HeuristicNLIVerifier.verify_claims_batch",
            "src.verification.verdict.classify_verification",
        ],
        "pipeline": pipeline_meta,
        "claim_level_output": pipeline_meta["claims"],
        "summary": {
            "claim_count": pipeline_meta["claim_count"],
            "generation_mode": pipeline_meta["generation_mode"],
            "retrieval_used": pipeline_meta["retrieval_used"],
            "retrieval_backend_status": pipeline_meta["retrieval_backend_status"],
            "retrieval_chunk_count": pipeline_meta["retrieval_chunk_count"],
            "verification_backend_status": pipeline_meta["verification_backend_status"],
            "llm_context_call_count": len(llm.context_calls),
            "retriever_query_count": len(retriever.queries),
        },
    }

    CLAIM_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLAIM_TRACE_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    assert CLAIM_TRACE_PATH.exists()
    assert pipeline_meta["claim_count"] > 0
    assert pipeline_meta["generation_mode"] == "rag"
    assert pipeline_meta["verification_backend_status"] == "ok"
    assert all("verification" in claim for claim in pipeline_meta["claims"])
    assert llm.context_calls
    assert retriever.queries


def _context_chunks_from_trace(source_trace: dict) -> list[LegalChunk]:
    document = source_trace["document"]
    raw_date = document.get("date_decided")
    parsed_date = date.fromisoformat(raw_date) if raw_date else None
    doc_id = str(document.get("id") or "ollama_trace_document")

    chunks = []
    for index, text in enumerate(source_trace["input"]["context"]):
        chunks.append(
            LegalChunk(
                id=f"{doc_id}_ollama_context_{index + 1}",
                doc_id=doc_id,
                text=text,
                chunk_index=index,
                doc_type="case",
                court_level=document.get("court_level"),
                citation=document.get("citation"),
                date_decided=parsed_date,
            )
        )
    return chunks
