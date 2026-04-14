"""Tests for retrieval index discovery."""

from __future__ import annotations

import json
from pathlib import Path

from src.indexing.index_discovery import discover_index_artifacts


def test_discover_index_artifacts_reads_suffix_summary(tmp_path: Path):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    summary_path = index_dir / "nli_100_index_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "bm25_path": "nli_100_bm25.pkl",
                "chroma_path": "nli_100_chroma",
                "collection_name": "nli_100_chunks",
            }
        ),
        encoding="utf-8",
    )

    artifacts = discover_index_artifacts(index_dir=index_dir)

    assert artifacts.summary_path == summary_path
    assert artifacts.bm25_path == (index_dir / "nli_100_bm25.pkl").resolve()
    assert artifacts.chroma_path == (index_dir / "nli_100_chroma").resolve()
    assert artifacts.collection_name == "nli_100_chunks"
    assert artifacts.source == "summary:nli_100_index_summary.json"


def test_discover_index_artifacts_prefers_canonical_summary(tmp_path: Path):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "nli_100_index_summary.json").write_text(
        json.dumps(
            {
                "bm25_path": "nli_100_bm25.pkl",
                "chroma_path": "nli_100_chroma",
                "collection_name": "nli_100_chunks",
            }
        ),
        encoding="utf-8",
    )
    canonical = index_dir / "index_summary.json"
    canonical.write_text(
        json.dumps(
            {
                "bm25_path": "bm25.pkl",
                "chroma_path": "chroma",
                "collection_name": "legal_chunks",
            }
        ),
        encoding="utf-8",
    )

    artifacts = discover_index_artifacts(index_dir=index_dir)

    assert artifacts.summary_path == canonical
    assert artifacts.bm25_path == canonical.parent / "bm25.pkl"
    assert artifacts.chroma_path == canonical.parent / "chroma"
    assert artifacts.collection_name == "legal_chunks"
