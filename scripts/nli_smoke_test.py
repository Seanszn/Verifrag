"""Run a deterministic corpus-backed NLI smoke test without external model downloads."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.nli_test_utils import (
    DEFAULT_QUERY,
    HeuristicNLIVerifier,
    build_corpus_backed_test_case,
    build_mixed_response_with_contradiction,
    claims_to_payload,
    dump_json,
)
from src.verification.nli_verifier import NLIVerifier


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a corpus-backed response against the current BM25 corpus.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="Query to retrieve supporting chunks for.",
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
        help="How many BM25 hits to retrieve.",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=3,
        help="Maximum number of evidence-backed sentences in the generated response.",
    )
    parser.add_argument(
        "--response",
        type=str,
        default=None,
        help="Optional explicit response text. If omitted, the script generates one from the corpus.",
    )
    parser.add_argument(
        "--include-contradiction",
        action="store_true",
        help="Append one intentionally contradicted sentence to exercise the negative path.",
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
        response_text=args.response,
    )

    response = test_case.response
    contradicted_sentence = None
    if args.include_contradiction:
        response, contradicted_sentence = build_mixed_response_with_contradiction(response)

    claims = list(build_corpus_backed_test_case(
        query=args.query,
        index_path=args.index_path,
        top_k=args.top_k,
        max_sentences=args.max_sentences,
        response_text=response,
    ).claims)

    verifier = HeuristicNLIVerifier() if args.verifier_mode == "heuristic" else NLIVerifier(device="cpu")
    chunks = [hit.chunk for hit in test_case.retrieved_chunks]
    verdicts = verifier.verify_claims_batch(claims, chunks)

    verdict_payload = []
    for claim, verdict in zip(claims, verdicts):
        verdict_payload.append(
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
    supported_count = sum(1 for verdict in verdicts if verdict.final_score >= 0.5 and not verdict.is_contradicted)
    payload = {
        "query": args.query,
        "response": response,
        "supporting_sentences": list(test_case.supporting_sentences),
        "contradicted_sentence": contradicted_sentence,
        "claims": claims_to_payload(claims),
        "retrieved_chunks": [hit.to_dict() for hit in test_case.retrieved_chunks],
        "verification_results": verdict_payload,
        "summary": {
            "claim_count": len(claims),
            "supported_count": supported_count,
            "contradicted_count": contradicted_count,
            "verifier_mode": args.verifier_mode,
        },
    }

    if not claims:
        raise RuntimeError("Smoke test produced no claims to verify.")
    if args.include_contradiction and contradicted_count == 0:
        raise RuntimeError("Expected at least one contradicted claim, but none were detected.")
    if not args.include_contradiction and contradicted_count != 0:
        raise RuntimeError(
            f"Expected all claims to remain supported for the generated response, but saw {contradicted_count} contradicted claim(s)."
        )
    if not args.include_contradiction and supported_count == 0:
        raise RuntimeError("Expected at least one supported claim, but none scored as supported.")

    print(dump_json(payload, output_path=args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
