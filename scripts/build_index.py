"""Build ChromaDB and BM25 indices from processed chunk JSONL files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import INDEX_DIR, PROCESSED_DIR, VECTOR_STORE
from src.indexing.index_builder import build_indices


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build BM25 and Chroma indices from processed chunk files.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help=f"Directory containing processed chunk JSONL files (default: {PROCESSED_DIR})",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.jsonl",
        help="Glob pattern used to select processed files inside the processed directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=INDEX_DIR,
        help=f"Directory where index artifacts will be written (default: {INDEX_DIR})",
    )
    parser.add_argument(
        "--bm25-name",
        type=str,
        default="bm25.pkl",
        help="Filename for the persisted BM25 artifact.",
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default="chroma",
        help="Subdirectory name for the persistent Chroma store.",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="index_summary.json",
        help="Filename for the build summary JSON.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=VECTOR_STORE.chroma_collection,
        help="Chroma collection name to populate.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    files = sorted(path for path in args.processed_dir.glob(args.pattern) if path.is_file())
    artifacts = build_indices(
        processed_dir=args.processed_dir,
        processed_files=files,
        output_dir=args.output_dir,
        bm25_filename=args.bm25_name,
        chroma_dirname=args.chroma_dir,
        summary_filename=args.summary_name,
        collection_name=args.collection_name,
    )

    print(f"Indexed {artifacts.chunk_count} chunks from {len(artifacts.processed_files)} file(s).")
    print(f"BM25: {artifacts.bm25_path}")
    print(f"Chroma: {artifacts.chroma_path}")
    print(f"Summary: {artifacts.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
