"""Tests for raw-to-processed corpus preparation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prepare_corpus import main as prepare_corpus_main
from src.ingestion.corpus_preparer import prepare_corpus


pytestmark = pytest.mark.smoke


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_raw_file(path: Path) -> None:
    rows = [
        {
            "id": "cl_1",
            "doc_type": "case",
            "case_name": "Example v. United States",
            "citation": "123 U.S. 456",
            "court": "scotus",
            "court_level": "scotus",
            "date_decided": "2026-03-01",
            "full_text": (
                "FACTS\n\n"
                "The officer entered the home without a warrant and arrested Doe.\n\n"
                "ANALYSIS\n\n"
                "The exclusionary rule requires suppression."
            ),
        },
        {
            "id": "cl_2",
            "doc_type": "case",
            "case_name": "Venue Corp. v. Transfer LLC",
            "citation": "789 F.3d 101",
            "court": "ca9",
            "court_level": "circuit",
            "date_decided": "2025-11-19",
            "full_text": "BACKGROUND\n\nTransfer under Section 1404 turns on convenience and justice.",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_prepare_corpus_writes_processed_chunks_and_summary(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_file = raw_dir / "scotus_cases.jsonl"
    _write_raw_file(raw_file)

    summary = prepare_corpus(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        chunk_size=12,
        chunk_overlap=3,
    )

    processed_file = processed_dir / "scotus_chunks.jsonl"
    processed_rows = _read_jsonl(processed_file)
    summary_payload = json.loads((processed_dir / "prep_summary.json").read_text(encoding="utf-8"))

    assert summary.documents_loaded == 2
    assert summary.chunks_written == len(processed_rows)
    assert processed_rows
    assert processed_rows[0]["doc_id"] == "cl_1"
    assert processed_rows[0]["case_name"] == "Example v. United States"
    assert processed_rows[0]["court"] == "scotus"
    assert processed_rows[0]["citation"] == "123 U.S. 456"
    assert processed_rows[0]["chunk_index"] == 0
    assert all("section_type" not in row for row in processed_rows)
    assert summary_payload["raw_files"] == ["scotus_cases.jsonl"]
    assert summary_payload["output_file"] == "scotus_chunks.jsonl"
    assert summary_payload["documents_loaded"] == 2


def test_prepare_corpus_cli_uses_expected_output_names(tmp_path: Path, capsys):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _write_raw_file(raw_dir / "ca9_cases.jsonl")

    exit_code = prepare_corpus_main(
        [
            "--raw-dir",
            str(raw_dir),
            "--processed-dir",
            str(processed_dir),
            "--chunk-size",
            "10",
            "--chunk-overlap",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Prepared 2 documents into" in captured.out
    assert "ca9_cases.jsonl -> ca9_chunks.jsonl" in captured.out
    assert (processed_dir / "ca9_chunks.jsonl").exists()
    assert (processed_dir / "prep_summary.json").exists()
