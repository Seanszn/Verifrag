"""Export a step-by-step NLI trace bundle for a longer corpus-backed response."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.nli_test_utils import (
    BM25Retriever,
    HeuristicNLIVerifier,
    StaticResponseLLM,
    build_corpus_backed_test_case,
    claims_to_payload,
)
from src.indexing.index_discovery import discover_index_artifacts
from src.pipeline import QueryPipeline
from src.storage.database import Database
from src.verification.nli_verifier import NLIVerifier


DEFAULT_TRACE_QUERY = (
    "How did the Supreme Court analyze the TikTok law's First Amendment challenge, "
    "what level of scrutiny did it apply, and why did it uphold the challenged provisions?"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a longer corpus-backed NLI trace and persist step-by-step artifacts.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_TRACE_QUERY,
        help="Query to probe against the active corpus.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help="Optional BM25 artifact path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="How many BM25 hits to retrieve.",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=6,
        help="Maximum number of evidence-backed sentences in the generated response.",
    )
    parser.add_argument(
        "--verifier-mode",
        choices=("heuristic", "live"),
        default="heuristic",
        help="Use the deterministic offline verifier or the real Hugging Face verifier.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "nli_trace" / "tiktok_first_amendment_complex",
        help="Directory where the trace bundle will be written.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_index_artifacts()
    test_case = build_corpus_backed_test_case(
        query=args.query,
        index_path=args.index_path,
        top_k=args.top_k,
        max_sentences=args.max_sentences,
    )
    claims = list(test_case.claims)
    chunks = [hit.chunk for hit in test_case.retrieved_chunks]
    verifier = HeuristicNLIVerifier() if args.verifier_mode == "heuristic" else NLIVerifier(device="cpu")

    response_path = output_dir / "01_response.txt"
    generated_path = output_dir / "02_generated_test_case.json"
    claims_path = output_dir / "03_claims.json"
    smoke_path = output_dir / "04_smoke_verification.json"
    pipeline_path = output_dir / "05_pipeline_result.json"
    manifest_path = output_dir / "00_manifest.json"
    readme_path = output_dir / "README.md"

    response_path.write_text(test_case.response + "\n", encoding="utf-8")
    _write_json(generated_path, test_case.to_dict())
    _write_json(
        claims_path,
        {
            "query": args.query,
            "response": test_case.response,
            "claim_count": len(claims),
            "claims": claims_to_payload(claims),
        },
    )

    smoke_payload = _build_smoke_payload(
        query=args.query,
        response=test_case.response,
        supporting_sentences=list(test_case.supporting_sentences),
        claims=claims,
        retrieved_chunks=[hit.to_dict() for hit in test_case.retrieved_chunks],
        verifier=verifier,
        chunks=chunks,
        verifier_mode=args.verifier_mode,
    )
    _write_json(smoke_path, smoke_payload)

    pipeline_payload = _build_pipeline_payload(
        query=args.query,
        response=test_case.response,
        index_path=args.index_path,
        top_k=args.top_k,
        verifier=verifier,
        verifier_mode=args.verifier_mode,
    )
    _write_json(pipeline_path, pipeline_payload)

    manifest = {
        "query": args.query,
        "output_dir": str(output_dir),
        "verifier_mode": args.verifier_mode,
        "top_k": args.top_k,
        "max_sentences": args.max_sentences,
        "active_index_artifacts": {
            "source": discovered.source,
            "summary_path": str(discovered.summary_path) if discovered.summary_path is not None else None,
            "bm25_path": str((args.index_path or discovered.bm25_path).resolve()),
            "chroma_path": str(discovered.chroma_path),
            "collection_name": discovered.collection_name,
        },
        "outputs": {
            "response_text": str(response_path),
            "generated_test_case": str(generated_path),
            "claims": str(claims_path),
            "smoke_verification": str(smoke_path),
            "pipeline_result": str(pipeline_path),
            "readme": str(readme_path),
        },
    }
    _write_json(manifest_path, manifest)
    readme_path.write_text(_build_readme(manifest), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    return 0


def _build_smoke_payload(
    *,
    query: str,
    response: str,
    supporting_sentences: list[str],
    claims: list[Any],
    retrieved_chunks: list[dict[str, Any]],
    verifier,
    chunks: list[Any],
    verifier_mode: str,
) -> dict[str, Any]:
    verdicts = verifier.verify_claims_batch(claims, chunks)
    verification_results = []
    for claim, verdict in zip(claims, verdicts):
        verification_results.append(
            {
                "claim": claim.to_dict(),
                "verification": {
                    "final_score": verdict.final_score,
                    "is_contradicted": verdict.is_contradicted,
                    "best_chunk_idx": verdict.best_chunk_idx,
                    "support_ratio": verdict.support_ratio,
                    "component_scores": verdict.component_scores,
                    "best_chunk": verdict.best_chunk.to_dict() if verdict.best_chunk is not None else None,
                },
            }
        )

    contradicted_count = sum(1 for verdict in verdicts if verdict.is_contradicted)
    supported_count = sum(
        1
        for verdict in verdicts
        if verdict.final_score >= 0.5 and not verdict.is_contradicted
    )

    return {
        "query": query,
        "response": response,
        "supporting_sentences": supporting_sentences,
        "claims": claims_to_payload(claims),
        "retrieved_chunks": retrieved_chunks,
        "verification_results": verification_results,
        "summary": {
            "claim_count": len(claims),
            "supported_count": supported_count,
            "contradicted_count": contradicted_count,
            "verifier_mode": verifier_mode,
        },
    }


def _build_pipeline_payload(
    *,
    query: str,
    response: str,
    index_path: Path | None,
    top_k: int,
    verifier,
    verifier_mode: str,
) -> dict[str, Any]:
    retriever = BM25Retriever(index_path=index_path, top_k=top_k)

    with tempfile.TemporaryDirectory(prefix="nli-trace-") as temp_dir:
        db = Database(Path(temp_dir) / "pipeline.db")
        db.initialize()
        user = db.create_user("nli_trace", "not-a-real-hash")

        pipeline = QueryPipeline(
            db=db,
            llm=StaticResponseLLM(response),
            retriever=retriever,
            verifier=verifier,
        )
        result = pipeline.run(user_id=user["id"], query=query)

    meta = result["pipeline"]
    claims = meta.get("claims", [])
    contradicted_count = sum(
        1
        for claim in claims
        if claim.get("verification", {}).get("is_contradicted")
    )
    supported_count = sum(
        1
        for claim in claims
        if claim.get("verification", {}).get("final_score", 0.0) >= 0.5
        and not claim.get("verification", {}).get("is_contradicted")
    )

    return {
        "query": query,
        "response": response,
        "pipeline": meta,
        "summary": {
            "claim_count": len(claims),
            "supported_count": supported_count,
            "contradicted_count": contradicted_count,
            "verifier_mode": verifier_mode,
        },
    }


def _build_readme(manifest: Mapping[str, Any]) -> str:
    outputs = manifest["outputs"]
    active_index = manifest["active_index_artifacts"]
    return "\n".join(
        [
            "# NLI Trace Bundle",
            "",
            f"Query: {manifest['query']}",
            f"Verifier mode: {manifest['verifier_mode']}",
            "",
            "## Active index artifacts",
            f"- Summary source: {active_index['source']}",
            f"- Summary file: {active_index['summary_path']}",
            f"- BM25 index: {active_index['bm25_path']}",
            f"- Chroma store: {active_index['chroma_path']}",
            f"- Chroma collection: {active_index['collection_name']}",
            "",
            "## Output files",
            f"- `00_manifest.json`: run metadata, index selection, and a map of all output paths.",
            f"- `01_response.txt`: the longer response that is being verified.",
            f"- `02_generated_test_case.json`: query, response, supporting sentences, extracted claims, and retrieved BM25 chunks.",
            f"- `03_claims.json`: the extracted claim list isolated from the broader test case payload.",
            f"- `04_smoke_verification.json`: direct claim-vs-evidence verification results from the NLI verifier.",
            f"- `05_pipeline_result.json`: the same response pushed through `QueryPipeline` with retrieval and verification metadata.",
            "",
            "## Paths",
            f"- Manifest: {outputs['manifest'] if 'manifest' in outputs else 'see 00_manifest.json'}",
            f"- Response: {outputs['response_text']}",
            f"- Generated test case: {outputs['generated_test_case']}",
            f"- Claims: {outputs['claims']}",
            f"- Smoke verification: {outputs['smoke_verification']}",
            f"- Pipeline result: {outputs['pipeline_result']}",
        ]
    ) + "\n"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
