"""Helpers for discovering the active retrieval index artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import INDEX_DIR, VECTOR_STORE


@dataclass(frozen=True)
class IndexArtifacts:
    """Resolved paths and settings for the active retrieval index."""

    bm25_path: Path
    chroma_path: Path
    collection_name: str
    summary_path: Path | None = None
    source: str = "default"


def discover_index_artifacts(index_dir: str | Path = INDEX_DIR) -> IndexArtifacts:
    """Resolve the BM25 file, Chroma directory, and collection to load."""

    index_root = Path(index_dir)
    summary_path = _discover_summary_path(index_root)
    summary = _load_summary(summary_path) if summary_path is not None else None

    bm25_path = _resolve_explicit_path(os.getenv("BM25_PATH"))
    if bm25_path is None:
        bm25_path = _summary_path(summary, "bm25_path", summary_path) or (index_root / "bm25.pkl")

    chroma_path = _resolve_explicit_path(os.getenv("CHROMA_PATH"))
    if chroma_path is None:
        chroma_path = _summary_path(summary, "chroma_path", summary_path) or VECTOR_STORE.chroma_path

    collection_name = (
        os.getenv("CHROMA_COLLECTION")
        or _summary_value(summary, "collection_name")
        or VECTOR_STORE.chroma_collection
    )

    source = "default"
    if summary_path is not None and summary is not None:
        source = f"summary:{summary_path.name}"
    if os.getenv("BM25_PATH") or os.getenv("CHROMA_PATH") or os.getenv("CHROMA_COLLECTION"):
        source = "env"

    return IndexArtifacts(
        bm25_path=bm25_path,
        chroma_path=chroma_path,
        collection_name=collection_name,
        summary_path=summary_path if summary is not None else None,
        source=source,
    )


def _discover_summary_path(index_dir: Path) -> Path | None:
    explicit = _resolve_explicit_path(os.getenv("INDEX_SUMMARY_PATH"))
    if explicit is not None:
        return explicit if explicit.exists() else None

    canonical = index_dir / "index_summary.json"
    if canonical.exists():
        return canonical

    candidates = sorted(
        path
        for path in index_dir.glob("*_index_summary.json")
        if path.is_file()
    )
    if not candidates:
        return None

    return max(candidates, key=_summary_sort_key)


def _summary_sort_key(path: Path) -> tuple[float, str]:
    try:
        return (path.stat().st_mtime, path.name)
    except OSError:
        return (0.0, path.name)


def _load_summary(summary_path: Path | None) -> dict[str, Any] | None:
    if summary_path is None or not summary_path.exists():
        return None

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None
    return payload


def _summary_path(
    summary: dict[str, Any] | None,
    key: str,
    summary_path: Path | None,
) -> Path | None:
    value = _summary_value(summary, key)
    if not value:
        return None

    resolved = Path(str(value))
    if resolved.is_absolute():
        return resolved
    if summary_path is not None:
        return (summary_path.parent / resolved).resolve()
    return resolved


def _summary_value(summary: dict[str, Any] | None, key: str) -> Any | None:
    if summary is None:
        return None
    return summary.get(key)


def _resolve_explicit_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    return Path(raw_path).expanduser()
