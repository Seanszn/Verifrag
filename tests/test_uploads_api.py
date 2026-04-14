"""Tests for the upload API route."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import dependencies, uploads
from src.api.main import app
from src.auth.local_auth import hash_password
from src.storage.database import Database


pytestmark = pytest.mark.smoke


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_upload_requires_authentication(monkeypatch, tmp_path: Path):
    test_db = Database(tmp_path / "test.db")
    test_db.initialize()
    monkeypatch.setattr(dependencies, "database", test_db)

    with TestClient(app) as client:
        response = client.post(
            "/api/uploads",
            files=[("files", ("motion.txt", b"Test upload body", "text/plain"))],
        )

    assert response.status_code == 401


def test_upload_ingests_text_file_to_user_workspace(monkeypatch, tmp_path: Path):
    test_db = Database(tmp_path / "test.db")
    test_db.initialize()
    monkeypatch.setattr(dependencies, "database", test_db)

    user = test_db.create_user("alice", hash_password("password123"))
    token = test_db.create_session(user["id"])

    monkeypatch.setattr(uploads, "USER_UPLOADS_ROOT", tmp_path / "uploads")

    with TestClient(app) as client:
        response = client.post(
            "/api/uploads",
            headers=_auth_headers(token),
            data={"conversation_id": "7", "is_privileged": "true"},
            files=[
                (
                    "files",
                    (
                        "motion.txt",
                        b"BACKGROUND\n\nPlaintiff alleges a breach.\n\nANALYSIS\n\nThe claim survives dismissal.",
                        "text/plain",
                    ),
                )
            ],
        )

    assert response.status_code == 201
    payload = response.json()
    assert payload["conversation_id"] == 7
    assert payload["files_uploaded"] == 1
    assert payload["documents_upserted"] == 1
    assert payload["chunks_upserted"] >= 1
    assert payload["files"][0]["filename"] == "motion.txt"
    assert payload["files"][0]["is_privileged"] is True

    user_root = tmp_path / "uploads" / f"user_{user['id']}"
    raw_path = user_root / "raw" / uploads.RAW_FILENAME
    processed_path = user_root / "processed" / uploads.PROCESSED_FILENAME
    stored_files = list((user_root / "files").glob("*.txt"))

    assert raw_path.exists()
    assert processed_path.exists()
    assert stored_files

    raw_rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    processed_rows = [
        json.loads(line) for line in processed_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    assert len(raw_rows) == 1
    assert raw_rows[0]["doc_type"] == "user_upload"
    assert raw_rows[0]["source_file"] == "motion.txt"
    assert raw_rows[0]["is_privileged"] is True
    assert processed_rows
    assert all(row["doc_id"] == raw_rows[0]["id"] for row in processed_rows)
