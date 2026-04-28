"""Prepare raw corpus JSONL files into processed chunk JSONL files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DIR, RAW_DIR, RETRIEVAL
from src.ingestion.corpus_preparer import find_raw_files, prepare_corpus


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk raw legal document JSONL files into processed corpus files.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Directory containing raw JSONL documents (default: {RAW_DIR})",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.jsonl",
        help="Glob pattern used to select raw files inside the raw directory.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help=f"Directory where processed chunk files will be written (default: {PROCESSED_DIR})",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="prep_summary.json",
        help="Filename for the preparation summary JSON.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=RETRIEVAL.chunk_size,
        help=f"Chunk size in words (default: {RETRIEVAL.chunk_size})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=RETRIEVAL.chunk_overlap,
        help=f"Chunk overlap in words (default: {RETRIEVAL.chunk_overlap})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    files = find_raw_files(args.raw_dir, pattern=args.pattern)
    summary = prepare_corpus(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        raw_files=files,
        summary_filename=args.summary_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print(f"Prepared {summary.documents_loaded} documents into {summary.chunks_written} chunks.")
    for prepared in summary.prepared_files:
        print(
            f"{prepared.raw_file.name} -> {prepared.output_file.name} "
            f"({prepared.documents_loaded} docs, {prepared.chunks_written} chunks)"
        )
    print(f"Summary: {summary.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
