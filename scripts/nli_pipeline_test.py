"""Run a pipeline-level NLI integration test using a corpus-backed synthetic response."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.nli_test_utils import (
    BM25Retriever,
    DEFAULT_QUERY,
    HeuristicNLIVerifier,
    StaticResponseLLM,
    build_corpus_backed_test_case,
    build_mixed_response_with_contradiction,
    dump_json,
)
from src.pipeline import QueryPipeline
from src.storage.database import Database
from src.verification.nli_verifier import NLIVerifier


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise QueryPipeline with a corpus-backed synthetic assistant response.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="Query to run through the pipeline.",
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
        default=8,
        help="How many BM25 hits the retriever should return.",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=3,
        help="Maximum number of evidence-backed sentences in the generated response.",
    )
    parser.add_argument(
        "--include-contradiction",
        action="store_true",
        help="Append one intentionally contradicted sentence before running the pipeline.",
    )
    parser.add_argument(
        "--verifier-mode",
        choices=("heuristic", "live"),
        default="heuristic",
        help="Use the deterministic offline verifier or the real Hugging Face verifier.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    test_case = build_corpus_backed_test_case(
        query=args.query,
        index_path=args.index_path,
        top_k=args.top_k,
        max_sentences=args.max_sentences,
    )

    response = test_case.response
    contradicted_sentence = None
    if args.include_contradiction:
        response, contradicted_sentence = build_mixed_response_with_contradiction(response)

    retriever = BM25Retriever(index_path=args.index_path, top_k=args.top_k)
    verifier = HeuristicNLIVerifier() if args.verifier_mode == "heuristic" else NLIVerifier(device="cpu")

    with tempfile.TemporaryDirectory(prefix="nli-pipeline-") as temp_dir:
        db = Database(Path(temp_dir) / "pipeline.db")
        db.initialize()
        user = db.create_user("nli_smoke", "not-a-real-hash")

        pipeline = QueryPipeline(
            db=db,
            llm=StaticResponseLLM(response),
            retriever=retriever,
            verifier=verifier,
        )
        result = pipeline.run(user_id=user["id"], query=args.query)

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

    if meta.get("verification_backend_status") != "ok":
        raise RuntimeError(
            f"Pipeline verification did not complete successfully: {meta.get('verification_backend_status')}"
        )
    if not claims:
        raise RuntimeError("Pipeline test produced no claims.")
    if args.include_contradiction and contradicted_count == 0:
        raise RuntimeError("Expected a contradicted claim in the pipeline output, but none were found.")
    if not args.include_contradiction and contradicted_count != 0:
        raise RuntimeError(
            f"Expected all generated claims to remain supported in pipeline output, but saw {contradicted_count} contradicted claim(s)."
        )
    if not args.include_contradiction and supported_count == 0:
        raise RuntimeError("Expected at least one supported claim in the pipeline output.")

    payload = {
        "query": args.query,
        "response": response,
        "contradicted_sentence": contradicted_sentence,
        "pipeline": meta,
        "summary": {
            "claim_count": len(claims),
            "supported_count": supported_count,
            "contradicted_count": contradicted_count,
            "verifier_mode": args.verifier_mode,
        },
    }
    print(dump_json(payload, output_path=args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
