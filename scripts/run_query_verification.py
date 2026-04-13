"""Run a live Ollama RAG query and write claim-level verification JSON."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.ollama_backend import OllamaBackend
from src.indexing.index_discovery import discover_index_artifacts
from src.pipeline import QueryPipeline
from src.storage.database import Database
from src.verification.nli_verifier import NLIVerifier


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "query_verification"


class _TopKRetriever:
    """Adapter that lets the CLI override retrieval depth without changing the pipeline."""

    def __init__(self, retriever: Any, top_k: int) -> None:
        self.retriever = retriever
        self.top_k = top_k

    def retrieve(self, query: str, k: int = 10):
        _ = k
        return self.retriever.retrieve(query, k=self.top_k)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a query through retrieval, Ollama generation, claim decomposition, "
            "and NLI verification, then write a JSON claim-validity report."
        ),
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Query text. You may quote it or pass it as separate words.",
    )
    parser.add_argument(
        "--query",
        dest="query_option",
        help="Query text. Overrides the positional query when provided.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path. Defaults to artifacts/query_verification/query_verification_<timestamp>.json.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override how many retrieved chunks are passed to Ollama and NLI verification.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model name. Defaults to LLM_MODEL or the project config.",
    )
    parser.add_argument(
        "--ollama-host",
        default=None,
        help="Ollama host URL. Defaults to OLLAMA_HOST or the project config.",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=int,
        default=120,
        help="Ollama request timeout in seconds.",
    )
    parser.add_argument(
        "--verifier-mode",
        choices=("live", "heuristic"),
        default="live",
        help="Use the real Hugging Face NLI verifier or a deterministic offline smoke-test verifier.",
    )
    parser.add_argument(
        "--nli-model",
        default=None,
        help="Hugging Face NLI model name. Defaults to the project config.",
    )
    parser.add_argument(
        "--nli-device",
        default=None,
        help='Torch device for NLI, such as "cpu" or "cuda". Defaults to auto-detection.',
    )
    parser.add_argument(
        "--nli-batch-size",
        type=int,
        default=8,
        help="NLI inference batch size.",
    )
    parser.add_argument(
        "--nli-max-length",
        type=int,
        default=512,
        help="Tokenizer max length for NLI premise/hypothesis pairs.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Also print the generated JSON report to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    query = _resolve_query(args)
    output_path = _resolve_output_path(args.output)

    if args.top_k is not None and args.top_k <= 0:
        raise SystemExit("--top-k must be greater than zero.")

    llm = OllamaBackend(
        model=args.model,
        host=args.ollama_host,
        timeout=args.ollama_timeout,
    )
    verifier = _build_verifier(args)

    with tempfile.TemporaryDirectory(prefix="query-verification-") as temp_dir:
        db = Database(Path(temp_dir) / "query_verification.db")
        db.initialize()
        user = db.create_user("query_verification_cli", "not-a-real-hash")

        pipeline = QueryPipeline(
            db=db,
            llm=llm,
            verifier=verifier,
        )
        if args.top_k is not None and pipeline.retriever is not None:
            pipeline.retriever = _TopKRetriever(pipeline.retriever, args.top_k)

        result = pipeline.run(user_id=user["id"], query=query)

    report = _build_report(
        query=query,
        result=result,
        output_path=output_path,
        args=args,
        llm=llm,
        verifier=verifier,
    )
    _write_json(output_path, report)

    if args.stdout:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(str(output_path))
    return 0


def _resolve_query(args: argparse.Namespace) -> str:
    query = args.query_option if args.query_option is not None else " ".join(args.query)
    query = " ".join(str(query).split())
    if not query:
        raise SystemExit(
            "Provide a query, for example: python scripts/run_query_verification.py "
            '"What did Miranda v. Arizona hold?"'
        )
    return query


def _resolve_output_path(raw_path: Path | None) -> Path:
    if raw_path is not None:
        return raw_path.expanduser().resolve()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return (DEFAULT_OUTPUT_DIR / f"query_verification_{timestamp}.json").resolve()


def _build_report(
    *,
    query: str,
    result: dict[str, Any],
    output_path: Path,
    args: argparse.Namespace,
    llm: OllamaBackend,
    verifier: NLIVerifier,
) -> dict[str, Any]:
    pipeline_meta = result["pipeline"]
    response = str(result["assistant_message"]["content"])
    claims = [_claim_report(claim, pipeline_meta) for claim in pipeline_meta.get("claims", [])]
    verdict_counts = Counter(
        claim["validity"]["verdict"]
        for claim in claims
        if claim.get("validity", {}).get("verdict") is not None
    )
    index_artifacts = discover_index_artifacts()

    return {
        "query": query,
        "response": response,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_path": str(output_path),
        "generation": {
            "llm_provider": pipeline_meta.get("llm_provider"),
            "llm_model": llm.model,
            "ollama_host": llm.host,
            "mode": pipeline_meta.get("generation_mode"),
            "status": pipeline_meta.get("llm_backend_status"),
        },
        "retrieval": {
            "used": pipeline_meta.get("retrieval_used"),
            "status": pipeline_meta.get("retrieval_backend_status"),
            "chunk_count": pipeline_meta.get("retrieval_chunk_count"),
            "top_k_override": args.top_k,
            "active_index": {
                "source": index_artifacts.source,
                "summary_path": str(index_artifacts.summary_path) if index_artifacts.summary_path else None,
                "bm25_path": str(index_artifacts.bm25_path),
                "chroma_path": str(index_artifacts.chroma_path),
                "collection_name": index_artifacts.collection_name,
            },
            "chunks": pipeline_meta.get("retrieved_chunks", []),
        },
        "verification": {
            "status": pipeline_meta.get("verification_backend_status"),
            "mode": args.verifier_mode,
            "nli_model": verifier.model_name,
            "nli_device": verifier.device,
            "claim_count": pipeline_meta.get("claim_count", len(claims)),
            "verdict_counts": dict(sorted(verdict_counts.items())),
            "error": pipeline_meta.get("verification_error"),
        },
        "claims": claims,
        "pipeline": pipeline_meta,
    }


def _claim_report(claim: dict[str, Any], pipeline_meta: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in claim.items() if key != "verification"}
    payload["validity"] = _validity_summary(
        claim.get("verification"),
        pipeline_meta,
    )
    return payload


def _validity_summary(
    verification: dict[str, Any] | None,
    pipeline_meta: dict[str, Any],
) -> dict[str, Any]:
    if not verification:
        verification_status = pipeline_meta.get("verification_backend_status")
        verification_error = pipeline_meta.get("verification_error") or {}
        return {
            "verdict": "NOT_VERIFIED",
            "reason": verification_error.get("message") or verification_status or "missing_verification",
            "error_type": verification_error.get("type"),
            "final_score": None,
            "is_contradicted": None,
            "support_ratio": None,
            "best_evidence": None,
        }

    best_chunk = verification.get("best_chunk")
    return {
        "verdict": verification.get("verdict"),
        "reason": verification.get("verdict_explanation"),
        "final_score": verification.get("final_score"),
        "is_contradicted": verification.get("is_contradicted"),
        "support_ratio": verification.get("support_ratio"),
        "component_scores": verification.get("component_scores", {}),
        "best_evidence": _best_evidence_summary(best_chunk),
    }


def _best_evidence_summary(best_chunk: dict[str, Any] | None) -> dict[str, Any] | None:
    if not best_chunk:
        return None
    return {
        "chunk_id": best_chunk.get("id"),
        "doc_id": best_chunk.get("doc_id"),
        "citation": best_chunk.get("citation"),
        "court_level": best_chunk.get("court_level"),
        "date_decided": best_chunk.get("date_decided"),
        "text_preview": best_chunk.get("text_preview") or str(best_chunk.get("text", ""))[:280],
    }


def _build_verifier(args: argparse.Namespace) -> NLIVerifier:
    if args.verifier_mode == "heuristic":
        from scripts.nli_test_utils import HeuristicNLIVerifier

        return HeuristicNLIVerifier()

    return NLIVerifier(
        model_name=args.nli_model,
        device=args.nli_device,
        batch_size=args.nli_batch_size,
        max_length=args.nli_max_length,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
