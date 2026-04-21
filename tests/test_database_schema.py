"""Tests for SQLite schema initialization and migration behavior."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.storage.database import Database


pytestmark = pytest.mark.smoke


def test_initialize_migrates_legacy_schema_to_add_erd_style_tables(tmp_path: Path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token_hash TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        );

        CREATE TABLE conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            metadata_json TEXT
        );

        INSERT INTO conversations (id, user_id, title, created_at, updated_at)
        VALUES (1, 7, 'Legacy thread', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00');
        """
    )
    conn.commit()
    conn.close()

    db = Database(db_path)
    db.initialize()

    with db.connect() as migrated:
        table_names = {
            row["name"]
            for row in migrated.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        assert {
            "users",
            "sessions",
            "conversations",
            "conversation_state",
            "interactions",
            "messages",
            "verified_claims",
            "contradictions",
            "interaction_citations",
            "claim_citation_links",
        }.issubset(table_names)

        user_columns = {
            row["name"] for row in migrated.execute("PRAGMA table_info(users)").fetchall()
        }
        message_columns = {
            row["name"] for row in migrated.execute("PRAGMA table_info(messages)").fetchall()
        }
        assert "email" in user_columns
        assert "interaction_id" in message_columns

        state_row = migrated.execute(
            "SELECT conversation_id, summary, state_json FROM conversation_state WHERE conversation_id = 1"
        ).fetchone()
        assert state_row is not None
        assert state_row["conversation_id"] == 1
        assert state_row["summary"] is None
        assert state_row["state_json"] is None
