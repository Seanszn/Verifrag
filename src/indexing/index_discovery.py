"""Helpers for finding persisted retrieval artifacts on disk."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.config import INDEX_DIR, VECTOR_STORE

DEFAULT_SUMMARY_NAME = "index_summary.json"
CUSTOM_SUMMARY_GLOB = "*_index_summary.json"


@dataclass(frozen=True)
class IndexArtifacts:
    bm25_path: Path
    chroma_path: Path
    collection_name: str
    summary_path: Path
    source: str


def discover_index_artifacts(index_dir: str | Path = INDEX_DIR) -> IndexArtifacts:
    root = Path(index_dir)
    discovered = _discover_summary_artifacts(root)
    if discovered is not None:
        return discovered

    return IndexArtifacts(
        bm25_path=(root / "bm25.pkl").resolve(),
        chroma_path=(root / "chroma").resolve(),
        collection_name=VECTOR_STORE.chroma_collection,
        summary_path=(root / DEFAULT_SUMMARY_NAME).resolve(),
        source="default:index_dir",
    )


def _resolve_artifact_path(root: Path, raw_value: object, fallback_name: str) -> Path:
    if isinstance(raw_value, str) and raw_value.strip():
        path = Path(raw_value)
        if not path.is_absolute():
            path = root / path
        return path.resolve()
    return (root / fallback_name).resolve()


def _discover_summary_artifacts(root: Path) -> IndexArtifacts | None:
    for summary_path in _candidate_summary_paths(root):
        payload = _load_summary_payload(summary_path)
        if payload is None:
            continue
        bm25_path = _resolve_artifact_path(root, payload.get("bm25_path"), "bm25.pkl")
        chroma_path = _resolve_artifact_path(root, payload.get("chroma_path"), "chroma")
        collection_name = str(payload.get("collection_name") or VECTOR_STORE.chroma_collection)
        return IndexArtifacts(
            bm25_path=bm25_path,
            chroma_path=chroma_path,
            collection_name=collection_name,
            summary_path=summary_path.resolve(),
            source=f"summary:{summary_path.name}",
        )
    return None


def _candidate_summary_paths(root: Path) -> list[Path]:
    default_summary = root / DEFAULT_SUMMARY_NAME
    candidates: list[Path] = []
    if default_summary.exists():
        candidates.append(default_summary)

    try:
        custom_summaries = [
            path
            for path in root.glob(CUSTOM_SUMMARY_GLOB)
            if path.is_file() and path.name != DEFAULT_SUMMARY_NAME
        ]
    except OSError:
        custom_summaries = []

    custom_summaries.sort(key=_custom_summary_sort_key)
    candidates.extend(custom_summaries)
    return candidates


def _custom_summary_sort_key(path: Path) -> tuple[int, str]:
    try:
        return (-path.stat().st_mtime_ns, path.name)
    except OSError:
        return (0, path.name)


def _load_summary_payload(summary_path: Path) -> dict | None:
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None
