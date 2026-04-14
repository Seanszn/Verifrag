"""Helpers for finding persisted retrieval artifacts on disk."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.config import INDEX_DIR, VECTOR_STORE


@dataclass(frozen=True)
class IndexArtifacts:
    bm25_path: Path
    chroma_path: Path
    collection_name: str
    summary_path: Path
    source: str


def discover_index_artifacts(index_dir: str | Path = INDEX_DIR) -> IndexArtifacts:
    root = Path(index_dir)
    summary_path = root / "index_summary.json"

    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            bm25_path = _resolve_artifact_path(root, payload.get("bm25_path"), "bm25.pkl")
            chroma_path = _resolve_artifact_path(root, payload.get("chroma_path"), "chroma")
            collection_name = str(payload.get("collection_name") or VECTOR_STORE.chroma_collection)
            return IndexArtifacts(
                bm25_path=bm25_path,
                chroma_path=chroma_path,
                collection_name=collection_name,
                summary_path=summary_path.resolve(),
                source="summary:index_summary.json",
            )

    return IndexArtifacts(
        bm25_path=(root / "bm25.pkl").resolve(),
        chroma_path=(root / "chroma").resolve(),
        collection_name=VECTOR_STORE.chroma_collection,
        summary_path=summary_path.resolve(),
        source="default:index_dir",
    )


def _resolve_artifact_path(root: Path, raw_value: object, fallback_name: str) -> Path:
    if isinstance(raw_value, str) and raw_value.strip():
        path = Path(raw_value)
        if not path.is_absolute():
            path = root / path
        return path.resolve()
    return (root / fallback_name).resolve()
