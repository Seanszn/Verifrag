"""
CLI for ingesting user-supplied files into the local corpus.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR, RAW_DIR, VECTOR_STORE
from src.ingestion.user_file_ingestion import UserFileCorpusIngestor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse local user files and add them to the corpus JSONL outputs.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more files or directories to ingest.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When an input is a directory, walk it recursively.",
    )
    parser.add_argument(
        "--privileged",
        action="store_true",
        help="Mark ingested files as privileged in the raw JSONL metadata.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Directory for raw JSONL output (default: {RAW_DIR}).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help=f"Directory for processed chunk JSONL output (default: {PROCESSED_DIR}).",
    )
    parser.add_argument(
        "--raw-output-file",
        default="user_uploads.jsonl",
        help="JSONL filename for raw uploaded documents.",
    )
    parser.add_argument(
        "--processed-output-file",
        default="user_upload_chunks.jsonl",
        help="JSONL filename for processed uploaded chunks.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild ChromaDB and BM25 after ingestion so the new files are searchable.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=VECTOR_STORE.chroma_batch_size,
        help="Batch size for embedding/index rebuild when --rebuild-index is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingestor = UserFileCorpusIngestor(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        raw_output_file=args.raw_output_file,
        processed_output_file=args.processed_output_file,
    )
    summary = ingestor.ingest_paths(
        args.inputs,
        recursive=args.recursive,
        is_privileged=args.privileged,
    )

    print(f"Files discovered: {summary.files_discovered}")
    print(f"Files ingested: {summary.files_ingested}")
    print(f"Documents upserted: {summary.documents_upserted}")
    print(f"Chunks upserted: {summary.chunks_upserted}")
    print(f"Raw output: {summary.raw_output_path}")
    print(f"Processed output: {summary.processed_output_path}")

    if not args.rebuild_index:
        print("Indices not rebuilt. Run `python scripts/build_index.py` to make uploads searchable.")
        return

    from scripts.build_index import build_indices, load_chunks_from_dir

    chunks = load_chunks_from_dir(args.processed_dir)
    if not chunks:
        raise SystemExit(f"No chunks found in {args.processed_dir} after ingestion.")

    total_chunks, vector_shape = build_indices(chunks, batch_size=max(1, args.batch_size))
    print(f"Indexed chunks: {total_chunks}")
    print(f"Embedding matrix shape: {vector_shape}")


if __name__ == "__main__":
    main()
