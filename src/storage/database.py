"""SQLite storage for users, sessions, conversations, messages, and interactions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from src.config import API, DATABASE


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    """Small SQLite wrapper for API persistence and normalized query history."""

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
                    email TEXT,
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

                CREATE TABLE IF NOT EXISTS conversation_state (
                    conversation_id INTEGER PRIMARY KEY,
                    summary TEXT,
                    last_updated_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    interaction_id INTEGER,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                    FOREIGN KEY (interaction_id) REFERENCES interactions(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS verified_claims (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interaction_id INTEGER NOT NULL,
                    claim_text TEXT NOT NULL,
                    verdict TEXT,
                    confidence REAL,
                    source_citation TEXT,
                    created_at TEXT NOT NULL,
                    claim_id TEXT,
                    claim_type TEXT,
                    claim_source TEXT,
                    certainty TEXT,
                    doc_section TEXT,
                    span_json TEXT,
                    metadata_json TEXT,
                    FOREIGN KEY (interaction_id) REFERENCES interactions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS contradictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interaction_id INTEGER NOT NULL,
                    chunk_i_id TEXT,
                    chunk_j_id TEXT,
                    score REAL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT,
                    FOREIGN KEY (interaction_id) REFERENCES interactions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS interaction_citations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interaction_id INTEGER NOT NULL,
                    doc_id TEXT,
                    chunk_id TEXT,
                    source_label TEXT,
                    score REAL,
                    used_in_prompt INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT,
                    FOREIGN KEY (interaction_id) REFERENCES interactions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
                ON conversations(user_id, updated_at DESC, id DESC);

                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id, id ASC);

                CREATE INDEX IF NOT EXISTS idx_interactions_conversation
                ON interactions(conversation_id, id ASC);

                CREATE INDEX IF NOT EXISTS idx_verified_claims_interaction
                ON verified_claims(interaction_id, id ASC);

                CREATE INDEX IF NOT EXISTS idx_contradictions_interaction
                ON contradictions(interaction_id, id ASC);

                CREATE INDEX IF NOT EXISTS idx_interaction_citations_interaction
                ON interaction_citations(interaction_id, id ASC);
                """
            )
            self._migrate_schema(conn)
            conn.executescript(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email
                ON users(email);

                CREATE INDEX IF NOT EXISTS idx_messages_interaction
                ON messages(interaction_id, id ASC);
                """
            )

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """Backfill additive columns and rows when opening an older database file."""

        user_columns = self._table_columns(conn, "users")
        if "email" not in user_columns:
            conn.execute("ALTER TABLE users ADD COLUMN email TEXT")

        message_columns = self._table_columns(conn, "messages")
        if "interaction_id" not in message_columns:
            conn.execute("ALTER TABLE messages ADD COLUMN interaction_id INTEGER")

        conn.execute(
            """
            INSERT INTO conversation_state (conversation_id, summary, last_updated_at)
            SELECT c.id, NULL, c.updated_at
            FROM conversations c
            WHERE NOT EXISTS (
                SELECT 1
                FROM conversation_state s
                WHERE s.conversation_id = c.id
            )
            """
        )

    def _table_columns(self, conn: sqlite3.Connection, table_name: str) -> set[str]:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {str(row["name"]) for row in rows}

    def _touch_conversation(
        self,
        conn: sqlite3.Connection,
        conversation_id: int,
        *,
        updated_at: str,
    ) -> None:
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (updated_at, conversation_id),
        )
        conn.execute(
            """
            INSERT INTO conversation_state (conversation_id, summary, last_updated_at)
            VALUES (?, NULL, ?)
            ON CONFLICT(conversation_id) DO UPDATE
            SET last_updated_at = excluded.last_updated_at
            """,
            (conversation_id, updated_at),
        )

    def create_user(
        self,
        username: str,
        password_hash: str,
        email: str | None = None,
    ) -> dict[str, Any]:
        created_at = utc_now_iso()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (username, email, password_hash, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (username, email, password_hash, created_at),
            )
            row = conn.execute(
                "SELECT id, username, email, created_at FROM users WHERE id = ?",
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
                "SELECT id, username, email, created_at FROM users WHERE id = ?",
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
                SELECT u.id, u.username, u.email, u.created_at, s.expires_at
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
            "email": row["email"],
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
            conn.execute(
                """
                INSERT INTO conversation_state (conversation_id, summary, last_updated_at)
                VALUES (?, NULL, ?)
                ON CONFLICT(conversation_id) DO UPDATE
                SET last_updated_at = excluded.last_updated_at
                """,
                (cursor.lastrowid, now),
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
        interaction_id: int | None = None,
        metadata_json: Optional[str] = None,
    ) -> dict[str, Any]:
        created_at = utc_now_iso()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO messages (
                    conversation_id,
                    interaction_id,
                    role,
                    content,
                    created_at,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    interaction_id,
                    role,
                    content,
                    created_at,
                    metadata_json,
                ),
            )
            self._touch_conversation(conn, conversation_id, updated_at=created_at)
            row = conn.execute(
                """
                SELECT id, conversation_id, interaction_id, role, content, created_at, metadata_json
                FROM messages
                WHERE id = ?
                """,
                (cursor.lastrowid,),
            ).fetchone()
        return dict(row)

    def create_interaction(self, conversation_id: int, query: str) -> dict[str, Any]:
        created_at = utc_now_iso()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO interactions (conversation_id, query, response, created_at)
                VALUES (?, ?, NULL, ?)
                """,
                (conversation_id, query, created_at),
            )
            self._touch_conversation(conn, conversation_id, updated_at=created_at)
            row = conn.execute(
                """
                SELECT id, conversation_id, query, response, created_at
                FROM interactions
                WHERE id = ?
                """,
                (cursor.lastrowid,),
            ).fetchone()
        return dict(row)

    def complete_interaction(self, interaction_id: int, response: str) -> dict[str, Any]:
        completed_at = utc_now_iso()
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT id, conversation_id, query, response, created_at
                FROM interactions
                WHERE id = ?
                """,
                (interaction_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Interaction {interaction_id} does not exist.")
            conn.execute(
                "UPDATE interactions SET response = ? WHERE id = ?",
                (response, interaction_id),
            )
            self._touch_conversation(conn, row["conversation_id"], updated_at=completed_at)
            updated = conn.execute(
                """
                SELECT id, conversation_id, query, response, created_at
                FROM interactions
                WHERE id = ?
                """,
                (interaction_id,),
            ).fetchone()
        return dict(updated)

    def update_conversation_state(self, conversation_id: int, summary: str | None) -> None:
        updated_at = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_state (conversation_id, summary, last_updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE
                SET summary = excluded.summary,
                    last_updated_at = excluded.last_updated_at
                """,
                (conversation_id, summary, updated_at),
            )
            self._touch_conversation(conn, conversation_id, updated_at=updated_at)

    def replace_verified_claims(
        self,
        interaction_id: int,
        claims: list[dict[str, Any]],
    ) -> None:
        created_at = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM verified_claims WHERE interaction_id = ?",
                (interaction_id,),
            )
            for claim in claims:
                verification = claim.get("verification") or {}
                best_chunk = verification.get("best_chunk") or {}
                conn.execute(
                    """
                    INSERT INTO verified_claims (
                        interaction_id,
                        claim_text,
                        verdict,
                        confidence,
                        source_citation,
                        created_at,
                        claim_id,
                        claim_type,
                        claim_source,
                        certainty,
                        doc_section,
                        span_json,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        interaction_id,
                        claim.get("text", ""),
                        verification.get("verdict"),
                        verification.get("final_score"),
                        best_chunk.get("citation") or best_chunk.get("case_name"),
                        created_at,
                        claim.get("claim_id"),
                        claim.get("claim_type"),
                        claim.get("source"),
                        claim.get("certainty"),
                        claim.get("doc_section"),
                        json.dumps(claim.get("span")) if claim.get("span") is not None else None,
                        json.dumps(claim),
                    ),
                )

    def replace_contradictions(
        self,
        interaction_id: int,
        contradictions: list[dict[str, Any]],
    ) -> None:
        created_at = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM contradictions WHERE interaction_id = ?",
                (interaction_id,),
            )
            for contradiction in contradictions:
                conn.execute(
                    """
                    INSERT INTO contradictions (
                        interaction_id,
                        chunk_i_id,
                        chunk_j_id,
                        score,
                        created_at,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        interaction_id,
                        contradiction.get("chunk_i_id"),
                        contradiction.get("chunk_j_id"),
                        contradiction.get("score"),
                        created_at,
                        json.dumps(contradiction),
                    ),
                )

    def replace_interaction_citations(
        self,
        interaction_id: int,
        citations: list[dict[str, Any]],
    ) -> None:
        created_at = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM interaction_citations WHERE interaction_id = ?",
                (interaction_id,),
            )
            for citation in citations:
                conn.execute(
                    """
                    INSERT INTO interaction_citations (
                        interaction_id,
                        doc_id,
                        chunk_id,
                        source_label,
                        score,
                        used_in_prompt,
                        created_at,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        interaction_id,
                        citation.get("doc_id"),
                        citation.get("chunk_id"),
                        citation.get("source_label"),
                        citation.get("score"),
                        1 if citation.get("used_in_prompt", True) else 0,
                        created_at,
                        json.dumps(citation),
                    ),
                )

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
                SELECT id, conversation_id, interaction_id, role, content, created_at, metadata_json
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id ASC
                """,
                (conversation_id,),
            ).fetchall()
        return [dict(row) for row in rows]
