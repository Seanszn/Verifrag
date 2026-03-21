"""SQLite storage for users, sessions, conversations, and messages."""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from src.config import API, DATABASE


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    """Small SQLite wrapper for API persistence."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or DATABASE.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                """
            )

    def create_user(self, username: str, password_hash: str) -> dict[str, Any]:
        created_at = utc_now_iso()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (username, password_hash, created_at)
                VALUES (?, ?, ?)
                """,
                (username, password_hash, created_at),
            )
            row = conn.execute(
                "SELECT id, username, created_at FROM users WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
        return dict(row)

    def get_user_by_username(self, username: str) -> Optional[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM users WHERE username = ?",
                (username,),
            ).fetchone()

    def get_user_by_id(self, user_id: int) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id, username, created_at FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def create_session(self, user_id: int) -> str:
        import secrets

        raw_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        created_at = datetime.now(timezone.utc)
        expires_at = created_at + timedelta(hours=API.token_ttl_hours)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (user_id, token_hash, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    user_id,
                    token_hash,
                    created_at.isoformat(),
                    expires_at.isoformat(),
                ),
            )
        return raw_token

    def get_user_for_token(self, token: str) -> Optional[dict[str, Any]]:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT u.id, u.username, u.created_at, s.expires_at
                FROM sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.token_hash = ?
                """,
                (token_hash,),
            ).fetchone()
        if not row:
            return None
        expires_at = datetime.fromisoformat(row["expires_at"])
        if expires_at <= datetime.now(timezone.utc):
            self.delete_session(token)
            return None
        return {
            "id": row["id"],
            "username": row["username"],
            "created_at": row["created_at"],
        }

    def delete_session(self, token: str) -> None:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self.connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))

    def create_conversation(self, user_id: int, title: str) -> dict[str, Any]:
        now = utc_now_iso()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO conversations (user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, title, now, now),
            )
            row = conn.execute(
                """
                SELECT id, user_id, title, created_at, updated_at
                FROM conversations
                WHERE id = ?
                """,
                (cursor.lastrowid,),
            ).fetchone()
        return dict(row)

    def get_conversation(self, conversation_id: int, user_id: int) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT id, user_id, title, created_at, updated_at
                FROM conversations
                WHERE id = ? AND user_id = ?
                """,
                (conversation_id, user_id),
            ).fetchone()
        return dict(row) if row else None

    def list_conversations(self, user_id: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, user_id, title, created_at, updated_at
                FROM conversations
                WHERE user_id = ?
                ORDER BY updated_at DESC, id DESC
                """,
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        metadata_json: Optional[str] = None,
    ) -> dict[str, Any]:
        created_at = utc_now_iso()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, role, content, created_at, metadata_json),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (created_at, conversation_id),
            )
            row = conn.execute(
                """
                SELECT id, conversation_id, role, content, created_at, metadata_json
                FROM messages
                WHERE id = ?
                """,
                (cursor.lastrowid,),
            ).fetchone()
        return dict(row)

    def list_messages(self, conversation_id: int, user_id: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            owner = conn.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            ).fetchone()
            if not owner:
                return []
            rows = conn.execute(
                """
                SELECT id, conversation_id, role, content, created_at, metadata_json
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id ASC
                """,
                (conversation_id,),
            ).fetchall()
        return [dict(row) for row in rows]
