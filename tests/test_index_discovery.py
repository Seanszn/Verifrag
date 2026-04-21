"""Tests for persisted index discovery."""

from __future__ import annotations

import json

from src.indexing.index_discovery import discover_index_artifacts


def _write_summary(path, *, bm25_path: str, chroma_path: str, collection_name: str) -> None:
    path.write_text(
        json.dumps(
            {
                "bm25_path": bm25_path,
                "chroma_path": chroma_path,
                "collection_name": collection_name,
            }
        ),
        encoding="utf-8",
    )


def test_discover_index_artifacts_prefers_default_summary_name(tmp_path):
    default_summary = tmp_path / "index_summary.json"
    custom_summary = tmp_path / "nli_100_index_summary.json"

    _write_summary(
        default_summary,
        bm25_path="bm25.pkl",
        chroma_path="chroma",
        collection_name="legal_chunks",
    )
    _write_summary(
        custom_summary,
        bm25_path="nli_100_bm25.pkl",
        chroma_path="nli_100_chroma",
        collection_name="nli_100_chunks",
    )

    artifacts = discover_index_artifacts(index_dir=tmp_path)

    assert artifacts.summary_path == default_summary.resolve()
    assert artifacts.bm25_path == (tmp_path / "bm25.pkl").resolve()
    assert artifacts.chroma_path == (tmp_path / "chroma").resolve()
    assert artifacts.collection_name == "legal_chunks"
    assert artifacts.source == "summary:index_summary.json"


def test_discover_index_artifacts_falls_back_to_custom_summary_names(tmp_path):
    summary_path = tmp_path / "nli_100_index_summary.json"
    _write_summary(
        summary_path,
        bm25_path="nli_100_bm25.pkl",
        chroma_path="nli_100_chroma",
        collection_name="nli_100_chunks",
    )

    artifacts = discover_index_artifacts(index_dir=tmp_path)

    assert artifacts.summary_path == summary_path.resolve()
    assert artifacts.bm25_path == (tmp_path / "nli_100_bm25.pkl").resolve()
    assert artifacts.chroma_path == (tmp_path / "nli_100_chroma").resolve()
    assert artifacts.collection_name == "nli_100_chunks"
    assert artifacts.source == "summary:nli_100_index_summary.json"


def test_discover_index_artifacts_skips_invalid_default_summary(tmp_path):
    (tmp_path / "index_summary.json").write_text("{not json", encoding="utf-8")
    summary_path = tmp_path / "nli_100_index_summary.json"
    _write_summary(
        summary_path,
        bm25_path="nli_100_bm25.pkl",
        chroma_path="nli_100_chroma",
        collection_name="nli_100_chunks",
    )

    artifacts = discover_index_artifacts(index_dir=tmp_path)

    assert artifacts.summary_path == summary_path.resolve()
    assert artifacts.source == "summary:nli_100_index_summary.json"
