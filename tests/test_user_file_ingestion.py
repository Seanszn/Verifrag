"""Tests for local user-file corpus ingestion."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.ingestion.user_file_ingestion import (
    UnsupportedUserFileError,
    UserFileCorpusIngestor,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _workspace_tmp_dir() -> tempfile.TemporaryDirectory[str]:
    base_dir = Path.cwd() / ".pytest_tmp"
    base_dir.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=base_dir)


def test_ingest_text_file_writes_raw_and_processed_jsonl():
    with _workspace_tmp_dir() as temp_dir:
        tmp_path = Path(temp_dir)
        source_file = tmp_path / "motion.txt"
        source_file.write_text(
            "BACKGROUND\n\n"
            "Plaintiff alleges the officer lacked probable cause.\n\n"
            "ANALYSIS\n\n"
            "Qualified immunity does not apply on these facts.",
            encoding="utf-8",
        )

        ingestor = UserFileCorpusIngestor(
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
        )
        summary = ingestor.ingest_path(source_file, is_privileged=True)

        assert summary.files_discovered == 1
        assert summary.files_ingested == 1
        assert summary.documents_upserted == 1
        assert summary.chunks_upserted >= 1

        raw_records = _read_jsonl(tmp_path / "raw" / "user_uploads.jsonl")
        processed_records = _read_jsonl(tmp_path / "processed" / "user_upload_chunks.jsonl")

        assert len(raw_records) == 1
        assert raw_records[0]["doc_type"] == "user_upload"
        assert raw_records[0]["source_file"] == "motion.txt"
        assert raw_records[0]["is_privileged"] is True
        assert raw_records[0]["id"] == summary.document_ids[0]

        assert processed_records
        assert all(record["doc_id"] == raw_records[0]["id"] for record in processed_records)
        assert processed_records[0]["doc_type"] == "user_upload"


def test_ingest_directory_skips_unsupported_files_and_deduplicates():
    with _workspace_tmp_dir() as temp_dir:
        tmp_path = Path(temp_dir)
        source_dir = tmp_path / "uploads"
        source_dir.mkdir()
        (source_dir / "brief.md").write_text("A short legal brief for testing.", encoding="utf-8")
        (source_dir / "notes.csv").write_text("ignored,data", encoding="utf-8")

        ingestor = UserFileCorpusIngestor(
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
        )

        first = ingestor.ingest_paths([source_dir])
        second = ingestor.ingest_paths([source_dir])

        raw_records = _read_jsonl(tmp_path / "raw" / "user_uploads.jsonl")

        assert first.files_discovered == 1
        assert second.files_discovered == 1
        assert len(raw_records) == 1


def test_ingest_rejects_unsupported_explicit_file():
    with _workspace_tmp_dir() as temp_dir:
        tmp_path = Path(temp_dir)
        source_file = tmp_path / "evidence.csv"
        source_file.write_text("a,b\n1,2\n", encoding="utf-8")

        ingestor = UserFileCorpusIngestor(
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
        )

        with pytest.raises(UnsupportedUserFileError):
            ingestor.ingest_path(source_file)
