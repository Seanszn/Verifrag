"""
Create a small embedding demo bundle from corpus JSONL documents.

Outputs:
  - embeddings.npy (full vectors)
  - metadata.json (doc metadata + preview dims)
  - preview.md (human-readable table)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add project root to import path when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indexing.embedder import Embedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate demo embedding artifacts.")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("data/raw_test_batch/scotus_cases.jsonl"),
        help="Source JSONL file containing legal documents.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/eval/embedding_demo"),
        help="Directory for demo output artifacts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of documents to embed.",
    )
    parser.add_argument(
        "--preview-dims",
        type=int,
        default=8,
        help="How many embedding dimensions to include in preview files.",
    )
    return parser.parse_args()


def load_docs(path: Path, limit: int) -> list[dict]:
    docs: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(docs) >= limit:
                break
            docs.append(json.loads(line))
    return docs


def write_preview_markdown(output_path: Path, rows: list[dict], preview_dims: int) -> None:
    headers = ["doc_id", "court", "date_decided", "norm", "first_dims"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        dim_values = ", ".join(f"{v:.4f}" for v in row["embedding_preview"][:preview_dims])
        lines.append(
            f"| {row['id']} | {row.get('court','')} | {row.get('date_decided','')} | "
            f"{row['norm']:.4f} | {dim_values} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    docs = load_docs(args.input_jsonl, args.limit)
    if not docs:
        raise SystemExit(f"No docs found in {args.input_jsonl}")

    texts = [d.get("full_text", "") for d in docs]
    embedder = Embedder()
    vectors = embedder.encode(texts, batch_size=min(16, max(1, len(texts))), normalize=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "embeddings.npy", vectors)

    rows = []
    for i, doc in enumerate(docs):
        row = {
            "id": doc.get("id"),
            "court": doc.get("court"),
            "date_decided": doc.get("date_decided"),
            "case_name": doc.get("case_name"),
            "embedding_preview": vectors[i, : args.preview_dims].tolist(),
            "norm": float(np.linalg.norm(vectors[i])),
        }
        rows.append(row)

    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    write_preview_markdown(args.output_dir / "preview.md", rows, args.preview_dims)

    print(f"Saved demo to: {args.output_dir}")
    print(f"Vectors shape: {vectors.shape}")
    print(f"Artifacts: embeddings.npy, metadata.json, preview.md")


if __name__ == "__main__":
    main()
