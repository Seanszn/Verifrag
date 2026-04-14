"""Generate a corpus-backed query/response pair for NLI smoke testing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.nli_test_utils import (
    DEFAULT_QUERY,
    build_corpus_backed_test_case,
    dump_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a test query/response pair from the current BM25-indexed corpus.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="Query to probe against the indexed corpus.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help="Optional BM25 artifact path. Defaults to data/index/nli_100_bm25.pkl when present.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="How many BM25 hits to inspect while generating the response.",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=3,
        help="Maximum number of evidence-backed sentences to include in the generated response.",
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

    payload = test_case.to_dict()
    text = dump_json(payload, output_path=args.output)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
