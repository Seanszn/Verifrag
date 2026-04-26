"""Run retrieval/generation coverage probes from a JSON evaluation set.

The default local mode exercises the real retrieval/prompt-selection pipeline with
a deterministic trace LLM, so it can produce evidence without a running Ollama
server. Use --mode api when the FastAPI backend is running and you want live
generation output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "eval" / "retrieval_generation_coverage_set_2026-04-26.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "test_reports" / "retrieval_generation_coverage_probe.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TraceLLM:
    """Deterministic LLM used to trace generation mode without external services."""

    model = "trace-llm-local-offline"
    model_name = model
    host = "local-offline"

    def generate_legal_answer(
        self,
        query: str,
        *,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> str:
        _ = conversation_history
        return (
            "No retrieved context was supplied for this local offline probe. "
            f"The query was: {query}"
        )

    def generate_with_context(
        self,
        query: str,
        context: list[str],
        max_tokens: int | None = None,
        *,
        conversation_history: list[dict[str, Any]] | None = None,
        case_posture: dict[str, Any] | None = None,
        response_depth: str = "concise",
        **_: Any,
    ) -> str:
        _ = max_tokens, conversation_history, case_posture
        preview = _preview_text(" ".join(context), limit=420) if context else "No context."
        return (
            f"Local offline {response_depth} RAG trace for: {query}\n\n"
            f"Retrieved context was supplied. Evidence preview: {preview}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("local-offline", "api"), default="local-offline")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--username", default=f"coverage_probe_{uuid4().hex[:8]}")
    parser.add_argument("--password", default="coverage_probe_password_123")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--id", action="append", default=[])
    parser.add_argument("--skip-verification", action="store_true")
    parser.add_argument("--include-responses", action="store_true")
    parser.add_argument("--request-timeout", type=int, default=60)
    parser.add_argument("--query-timeout", type=int, default=420)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = _load_dataset(args.dataset)
    items = _select_items(dataset.get("items", []), args)
    if not items:
        raise SystemExit("No dataset items matched the requested filters.")

    started = time.perf_counter()
    if args.mode == "api":
        run_payload = _run_api(dataset, items, args)
    else:
        run_payload = _run_local_offline(dataset, items, args)

    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    run_payload["summary"]["elapsed_ms"] = elapsed_ms
    run_payload["summary"]["output"] = str(args.output.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
    print(json.dumps(run_payload["summary"], indent=2))
    print(str(args.output.resolve()))
    return 0


def _load_dataset(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("items"), list):
        raise SystemExit(f"Dataset has no items list: {path}")
    return payload


def _select_items(items: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = [item for item in items if isinstance(item, dict) and item.get("query")]
    if args.category:
        categories = set(args.category)
        selected = [item for item in selected if item.get("category") in categories]
    if args.id:
        ids = set(args.id)
        selected = [item for item in selected if item.get("id") in ids]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def _run_local_offline(
    dataset: dict[str, Any],
    items: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from src.indexing.bm25_index import BM25Index
    from src.ingestion.chunker import chunk_document
    from src.ingestion.document import LegalDocument
    import src.pipeline as pipeline_module
    from src.pipeline import QueryPipeline
    from src.retrieval import user_uploads
    from src.storage.database import Database
    from src.verification.heuristic_verifier import HeuristicNLIVerifier

    with tempfile.TemporaryDirectory(prefix="rgc-local-") as temp_dir:
        temp_root = Path(temp_dir)
        db = Database(temp_root / "coverage_probe.db")
        db.initialize()
        user = db.create_user(f"local_coverage_{uuid4().hex[:8]}", "not-a-real-hash")

        uploads_root = temp_root / "uploads"
        upload_setup = {"status": "not_requested", "fixtures": []}
        if any(item.get("include_uploaded_chunks") for item in items):
            upload_setup = _prepare_local_uploads(
                dataset,
                user_id=int(user["id"]),
                uploads_root=uploads_root,
                LegalDocument=LegalDocument,
                chunk_document=chunk_document,
                BM25Index=BM25Index,
            )

        original_loader = pipeline_module.load_user_upload_retriever

        def _load_temp_upload_retriever(user_id: int, *, shared_embedder=None):
            return user_uploads.load_user_upload_retriever(
                user_id,
                uploads_root=uploads_root,
                shared_embedder=shared_embedder,
            )

        if upload_setup["status"] == "ok":
            pipeline_module.load_user_upload_retriever = _load_temp_upload_retriever

        try:
            pipeline = QueryPipeline(
                db=db,
                llm=TraceLLM(),
                verifier=HeuristicNLIVerifier(),
                enable_verification=not args.skip_verification,
            )
            results = _run_local_items(
                pipeline=pipeline,
                user_id=int(user["id"]),
                items=items,
                include_responses=args.include_responses,
            )
        finally:
            pipeline_module.load_user_upload_retriever = original_loader

    return _build_run_payload(
        dataset=dataset,
        items=items,
        results=results,
        mode=args.mode,
        upload_setup=upload_setup,
    )


def _prepare_local_uploads(
    dataset: dict[str, Any],
    *,
    user_id: int,
    uploads_root: Path,
    LegalDocument: Any,
    chunk_document: Any,
    BM25Index: Any,
) -> dict[str, Any]:
    user_root = uploads_root / f"user_{user_id}"
    processed_dir = user_root / "processed"
    index_dir = user_root / "index"
    files_dir = user_root / "files"
    processed_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    files_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    fixture_reports = []
    for fixture in dataset.get("upload_fixtures", []):
        if not isinstance(fixture, dict):
            continue
        fixture_id = str(fixture["id"])
        filename = str(fixture["filename"])
        text = str(fixture["text"])
        source_path = files_dir / filename
        source_path.write_text(text, encoding="utf-8")
        document = LegalDocument(
            id=f"upload_{fixture_id}",
            doc_type="user_upload",
            full_text=text,
            case_name=fixture_id.replace("_", " ").title(),
            source_file=filename,
            is_privileged=bool(fixture.get("is_privileged")),
        )
        chunks = chunk_document(document)
        all_chunks.extend(chunks)
        fixture_reports.append(
            {
                "id": fixture_id,
                "filename": filename,
                "is_privileged": bool(fixture.get("is_privileged")),
                "chunk_count": len(chunks),
            }
        )

    processed_path = processed_dir / "user_upload_chunks.jsonl"
    with processed_path.open("w", encoding="utf-8") as handle:
        for chunk in all_chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    bm25 = BM25Index(all_chunks, index_path=index_dir / "bm25.pkl")
    bm25.save()
    (index_dir / "index_summary.json").write_text(
        json.dumps(
            {
                "processed_dir": str(processed_dir),
                "processed_files": [str(processed_path)],
                "chunk_count": len(all_chunks),
                "embedding_shape": [0, 0],
                "embedding_model": "not_built:local_sparse_upload_overlay",
                "chroma_path": str(index_dir / "chroma"),
                "bm25_path": str(index_dir / "bm25.pkl"),
                "collection_name": f"user_{user_id}_uploads",
                "built_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"status": "ok", "uploads_root": str(uploads_root), "fixtures": fixture_reports}


def _run_local_items(
    *,
    pipeline: Any,
    user_id: int,
    items: list[dict[str, Any]],
    include_responses: bool,
) -> list[dict[str, Any]]:
    conversation_ids: dict[str, int] = {}
    results = []
    for item in items:
        conversation_key = str(item.get("conversation_key") or item["id"])
        conversation_id = conversation_ids.get(conversation_key)
        started = time.perf_counter()
        try:
            payload = pipeline.run(
                user_id=user_id,
                query=str(item["query"]),
                conversation_id=conversation_id,
                include_uploaded_chunks=bool(item.get("include_uploaded_chunks")),
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            conversation_ids[conversation_key] = int(payload["conversation"]["id"])
            results.append(
                _result_from_payload(
                    item,
                    payload,
                    elapsed_ms=elapsed_ms,
                    include_responses=include_responses,
                    error=None,
                )
            )
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            results.append(
                {
                    "item": _item_summary(item),
                    "elapsed_ms": elapsed_ms,
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "pipeline_summary": {},
                    "retrieval_evidence": [],
                    "checks": {"status": "error"},
                }
            )
    return results


def _run_api(
    dataset: dict[str, Any],
    items: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    base_url = args.base_url.rstrip("/")
    session = requests.Session()
    session.post(
        f"{base_url}/api/auth/register",
        json={"username": args.username, "password": args.password},
        timeout=args.request_timeout,
    )
    login = session.post(
        f"{base_url}/api/auth/login",
        json={"username": args.username, "password": args.password},
        timeout=args.request_timeout,
    )
    login.raise_for_status()
    token = login.json()["token"]
    session.headers.update({"Authorization": f"Bearer {token}"})

    upload_setup = {"status": "not_requested", "fixtures": []}
    if any(item.get("include_uploaded_chunks") for item in items):
        upload_setup = _upload_api_fixtures(session, base_url, dataset, args.request_timeout)

    conversation_ids: dict[str, int] = {}
    results = []
    for item in items:
        conversation_key = str(item.get("conversation_key") or item["id"])
        if conversation_key not in conversation_ids:
            conversation_ids[conversation_key] = _create_api_conversation(
                session,
                base_url,
                title=f"Coverage {conversation_key}",
                timeout=args.request_timeout,
            )

        started = time.perf_counter()
        try:
            response = session.post(
                f"{base_url}/api/query",
                json={
                    "query": item["query"],
                    "conversation_id": conversation_ids[conversation_key],
                    "include_uploaded_chunks": bool(item.get("include_uploaded_chunks")),
                },
                timeout=args.query_timeout,
            )
            response.raise_for_status()
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            results.append(
                _result_from_payload(
                    item,
                    response.json(),
                    elapsed_ms=elapsed_ms,
                    include_responses=args.include_responses,
                    error=None,
                )
            )
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            results.append(
                {
                    "item": _item_summary(item),
                    "elapsed_ms": elapsed_ms,
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "pipeline_summary": {},
                    "retrieval_evidence": [],
                    "checks": {"status": "error"},
                }
            )

    return _build_run_payload(
        dataset=dataset,
        items=items,
        results=results,
        mode=args.mode,
        upload_setup=upload_setup,
        base_url=base_url,
        username=args.username,
    )


def _upload_api_fixtures(
    session: requests.Session,
    base_url: str,
    dataset: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    fixture_reports = []
    for fixture in dataset.get("upload_fixtures", []):
        if not isinstance(fixture, dict):
            continue
        files = [
            (
                "files",
                (
                    str(fixture["filename"]),
                    str(fixture["text"]).encode("utf-8"),
                    "text/plain",
                ),
            )
        ]
        response = session.post(
            f"{base_url}/api/uploads",
            data={"is_privileged": "true" if fixture.get("is_privileged") else "false"},
            files=files,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        fixture_reports.append(
            {
                "id": fixture.get("id"),
                "filename": fixture.get("filename"),
                "is_privileged": bool(fixture.get("is_privileged")),
                "chunks_upserted": payload.get("chunks_upserted"),
                "documents_upserted": payload.get("documents_upserted"),
            }
        )
    return {"status": "ok", "fixtures": fixture_reports}


def _create_api_conversation(
    session: requests.Session,
    base_url: str,
    *,
    title: str,
    timeout: int,
) -> int:
    response = session.post(f"{base_url}/api/conversations", json={"title": title}, timeout=timeout)
    response.raise_for_status()
    return int(response.json()["id"])


def _result_from_payload(
    item: dict[str, Any],
    payload: dict[str, Any],
    *,
    elapsed_ms: float,
    include_responses: bool,
    error: str | None,
) -> dict[str, Any]:
    pipeline = payload.get("pipeline") or {}
    assistant_content = str(payload.get("assistant_message", {}).get("content", ""))
    result = {
        "item": _item_summary(item),
        "conversation_id": payload.get("conversation", {}).get("id"),
        "interaction_id": payload.get("interaction", {}).get("id"),
        "elapsed_ms": elapsed_ms,
        "error": error,
        "pipeline_summary": _pipeline_summary(pipeline),
        "retrieval_evidence": _retrieval_evidence(pipeline),
        "checks": _evaluate_expected_signals(item, assistant_content, pipeline),
    }
    if include_responses:
        result["assistant_content"] = assistant_content
    return result


def _item_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("id"),
        "category": item.get("category"),
        "coverage": item.get("coverage", []),
        "query": item.get("query"),
        "include_uploaded_chunks": bool(item.get("include_uploaded_chunks")),
        "upload_fixture_id": item.get("upload_fixture_id"),
        "expected_relevant_cases": item.get("expected_relevant_cases", []),
        "expected_relevant_terms": item.get("expected_relevant_terms", []),
        "expected_generation": item.get("expected_generation"),
    }


def _pipeline_summary(pipeline: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "llm_backend_status",
        "generation_mode",
        "response_depth",
        "retrieval_used",
        "retrieval_backend_status",
        "public_retrieval_backend_status",
        "user_upload_retrieval_backend_status",
        "retrieval_chunk_count",
        "public_retrieval_chunk_count",
        "user_upload_retrieval_chunk_count",
        "prompt_chunk_count",
        "prompt_case_filter_status",
        "target_case_name",
        "query_grounding_status",
        "generation_context_status",
        "research_leads_status",
        "case_posture_status",
        "verification_backend_status",
        "verification_chunk_count",
        "verification_scope_status",
        "claim_count",
        "claim_support_summary",
        "timings_ms",
    ]
    return {key: pipeline.get(key) for key in keys if key in pipeline}


def _retrieval_evidence(pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = pipeline.get("retrieved_chunks") or []
    evidence = []
    for rank, chunk in enumerate(chunks[:8], start=1):
        if not isinstance(chunk, dict):
            continue
        evidence.append(
            {
                "rank": rank,
                "chunk_id": chunk.get("id"),
                "doc_id": chunk.get("doc_id"),
                "doc_type": chunk.get("doc_type"),
                "case_name": chunk.get("case_name"),
                "citation": chunk.get("citation"),
                "court_level": chunk.get("court_level"),
                "date_decided": chunk.get("date_decided"),
                "source_file": chunk.get("source_file"),
                "text_preview": _preview_text(str(chunk.get("text") or ""), limit=260),
            }
        )
    return evidence


def _evaluate_expected_signals(
    item: dict[str, Any],
    assistant_content: str,
    pipeline: dict[str, Any],
) -> dict[str, Any]:
    chunks = [chunk for chunk in pipeline.get("retrieved_chunks", []) if isinstance(chunk, dict)]
    retrieved_cases = _unique_values(chunk.get("case_name") for chunk in chunks)
    retrieved_source_files = _unique_values(chunk.get("source_file") for chunk in chunks)
    retrieved_text = " ".join(
        " ".join(
            str(chunk.get(field) or "")
            for field in ("case_name", "citation", "source_file", "text")
        )
        for chunk in chunks
    )

    expected_cases = [str(case) for case in item.get("expected_relevant_cases", [])]
    expected_terms = [str(term) for term in item.get("expected_relevant_terms", [])]
    forbidden_terms = [str(term) for term in item.get("forbidden_terms", [])]
    expected_upload = item.get("expected_upload_source_file")

    matched_cases = [
        case
        for case in expected_cases
        if any(_normalized(case) in _normalized(retrieved) for retrieved in retrieved_cases)
    ]
    matched_terms = [
        term
        for term in expected_terms
        if _normalized(term) in _normalized(retrieved_text)
        or _normalized(term) in _normalized(assistant_content)
    ]
    forbidden_hits = [
        term for term in forbidden_terms if _normalized(term) in _normalized(assistant_content)
    ]

    upload_match = None
    if expected_upload:
        upload_match = expected_upload in retrieved_source_files

    return {
        "matched_expected_cases": matched_cases,
        "missing_expected_cases": [case for case in expected_cases if case not in matched_cases],
        "matched_expected_terms": matched_terms,
        "missing_expected_terms": [term for term in expected_terms if term not in matched_terms],
        "expected_upload_source_file": expected_upload,
        "upload_source_file_match": upload_match,
        "forbidden_answer_hits": forbidden_hits,
        "retrieved_case_names": retrieved_cases,
        "retrieved_source_files": retrieved_source_files,
        "status": "ok" if not forbidden_hits else "warning:forbidden_answer_terms",
    }


def _build_run_payload(
    *,
    dataset: dict[str, Any],
    items: list[dict[str, Any]],
    results: list[dict[str, Any]],
    mode: str,
    upload_setup: dict[str, Any],
    base_url: str | None = None,
    username: str | None = None,
) -> dict[str, Any]:
    category_counts = Counter(str(item.get("category", "unknown")) for item in items)
    coverage_counts = Counter(
        str(coverage)
        for item in items
        for coverage in item.get("coverage", [])
    )
    retrieval_status_counts = Counter(
        str(result.get("pipeline_summary", {}).get("retrieval_backend_status", "missing"))
        for result in results
    )
    generation_mode_counts = Counter(
        str(result.get("pipeline_summary", {}).get("generation_mode", "missing"))
        for result in results
    )
    prompt_filter_counts = Counter(
        str(result.get("pipeline_summary", {}).get("prompt_case_filter_status", "missing"))
        for result in results
    )
    verification_status_counts = Counter(
        str(result.get("pipeline_summary", {}).get("verification_backend_status", "missing"))
        for result in results
    )
    errors = [result for result in results if result.get("error")]
    missing_expected_cases = defaultdict(list)
    missing_upload_matches = []
    forbidden_hits = []
    for result in results:
        checks = result.get("checks", {})
        item_id = result.get("item", {}).get("id")
        for case in checks.get("missing_expected_cases", []):
            missing_expected_cases[case].append(item_id)
        if checks.get("upload_source_file_match") is False:
            missing_upload_matches.append(item_id)
        if checks.get("forbidden_answer_hits"):
            forbidden_hits.append(
                {
                    "id": item_id,
                    "hits": checks.get("forbidden_answer_hits"),
                }
            )

    summary = {
        "dataset_name": dataset.get("name"),
        "mode": mode,
        "base_url": base_url,
        "username": username,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "item_count": len(results),
        "error_count": len(errors),
        "category_counts": dict(sorted(category_counts.items())),
        "coverage_counts": dict(sorted(coverage_counts.items())),
        "retrieval_status_counts": dict(sorted(retrieval_status_counts.items())),
        "generation_mode_counts": dict(sorted(generation_mode_counts.items())),
        "prompt_filter_status_counts": dict(sorted(prompt_filter_counts.items())),
        "verification_status_counts": dict(sorted(verification_status_counts.items())),
        "missing_expected_cases": dict(sorted(missing_expected_cases.items())),
        "missing_upload_match_item_ids": missing_upload_matches,
        "forbidden_answer_hits": forbidden_hits,
        "upload_setup": upload_setup,
    }
    return {
        "dataset": {
            "name": dataset.get("name"),
            "created_on": dataset.get("created_on"),
            "description": dataset.get("description"),
        },
        "summary": summary,
        "results": results,
    }


def _unique_values(values: Any) -> list[str]:
    unique = []
    seen = set()
    for value in values:
        if value is None:
            continue
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _normalized(text: str) -> str:
    return " ".join(str(text).lower().split())


def _preview_text(text: str, *, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


if __name__ == "__main__":
    raise SystemExit(main())
