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

                CREATE TABLE IF NOT EXISTS claim_citation_links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    verified_claim_id INTEGER NOT NULL,
                    interaction_citation_id INTEGER NOT NULL,
                    relationship TEXT NOT NULL,
                    score REAL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT,
                    FOREIGN KEY (verified_claim_id) REFERENCES verified_claims(id) ON DELETE CASCADE,
                    FOREIGN KEY (interaction_citation_id) REFERENCES interaction_citations(id) ON DELETE CASCADE
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

                CREATE INDEX IF NOT EXISTS idx_claim_citation_links_claim
                ON claim_citation_links(verified_claim_id, relationship, id ASC);
                """
            )
            self._migrate_schema(conn)
            conn.executescript(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email
                ON users(email);

                CREATE INDEX IF NOT EXISTS idx_messages_interaction
                ON messages(interaction_id, id ASC);

                CREATE UNIQUE INDEX IF NOT EXISTS idx_verified_claims_interaction_claim_id
                ON verified_claims(interaction_id, claim_id)
                WHERE claim_id IS NOT NULL;

                CREATE UNIQUE INDEX IF NOT EXISTS idx_interaction_citations_interaction_chunk
                ON interaction_citations(interaction_id, chunk_id)
                WHERE chunk_id IS NOT NULL;

                CREATE UNIQUE INDEX IF NOT EXISTS idx_claim_citation_links_unique
                ON claim_citation_links(verified_claim_id, interaction_citation_id, relationship);
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

    def list_interactions(self, conversation_id: int, user_id: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            owner = conn.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            ).fetchone()
            if not owner:
                return []
            rows = conn.execute(
                """
                SELECT id, conversation_id, query, response, created_at
                FROM interactions
                WHERE conversation_id = ?
                ORDER BY id ASC
                """,
                (conversation_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_interaction(self, interaction_id: int, user_id: int) -> Optional[dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT i.id, i.conversation_id, i.query, i.response, i.created_at
                FROM interactions i
                JOIN conversations c ON c.id = i.conversation_id
                WHERE i.id = ? AND c.user_id = ?
                """,
                (interaction_id, user_id),
            ).fetchone()
        return dict(row) if row else None

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

    def persist_interaction_artifacts(
        self,
        interaction_id: int,
        *,
        claims: list[dict[str, Any]],
        citations: list[dict[str, Any]],
        contradictions: list[dict[str, Any]],
    ) -> None:
        with self.connect() as conn:
            inserted_claims = self._replace_verified_claims(conn, interaction_id, claims)
            self._replace_contradictions(conn, interaction_id, contradictions)
            inserted_citations = self._replace_interaction_citations(conn, interaction_id, citations)
            self._replace_claim_citation_links(conn, inserted_claims, inserted_citations, claims)

    def replace_verified_claims(
        self,
        interaction_id: int,
        claims: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return self._replace_verified_claims(conn, interaction_id, claims)

    def replace_contradictions(
        self,
        interaction_id: int,
        contradictions: list[dict[str, Any]],
    ) -> None:
        with self.connect() as conn:
            self._replace_contradictions(conn, interaction_id, contradictions)

    def replace_interaction_citations(
        self,
        interaction_id: int,
        citations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return self._replace_interaction_citations(conn, interaction_id, citations)

    def replace_claim_citation_links(
        self,
        claim_rows: list[dict[str, Any]],
        citation_rows: list[dict[str, Any]],
        claims: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return self._replace_claim_citation_links(conn, claim_rows, citation_rows, claims)

    def _replace_verified_claims(
        self,
        conn: sqlite3.Connection,
        interaction_id: int,
        claims: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        created_at = utc_now_iso()
        conn.execute(
            "DELETE FROM verified_claims WHERE interaction_id = ?",
            (interaction_id,),
        )

        inserted_claims: list[dict[str, Any]] = []
        for claim in claims:
            verification = claim.get("verification") or {}
            best_chunk = (
                verification.get("best_supporting_chunk")
                or verification.get("best_chunk")
                or {}
            )
            cursor = conn.execute(
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
            row = conn.execute(
                """
                SELECT
                    id,
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
                FROM verified_claims
                WHERE id = ?
                """,
                (cursor.lastrowid,),
            ).fetchone()
            inserted_claims.append(dict(row))
        return inserted_claims

    def _replace_contradictions(
        self,
        conn: sqlite3.Connection,
        interaction_id: int,
        contradictions: list[dict[str, Any]],
    ) -> None:
        created_at = utc_now_iso()
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

    def _replace_interaction_citations(
        self,
        conn: sqlite3.Connection,
        interaction_id: int,
        citations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        created_at = utc_now_iso()
        conn.execute(
            "DELETE FROM interaction_citations WHERE interaction_id = ?",
            (interaction_id,),
        )

        inserted_citations: list[dict[str, Any]] = []
        for citation in citations:
            cursor = conn.execute(
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
            row = conn.execute(
                """
                SELECT
                    id,
                    interaction_id,
                    doc_id,
                    chunk_id,
                    source_label,
                    score,
                    used_in_prompt,
                    created_at,
                    metadata_json
                FROM interaction_citations
                WHERE id = ?
                """,
                (cursor.lastrowid,),
            ).fetchone()
            inserted_citations.append(dict(row))
        return inserted_citations

    def _replace_claim_citation_links(
        self,
        conn: sqlite3.Connection,
        claim_rows: list[dict[str, Any]],
        citation_rows: list[dict[str, Any]],
        claims: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        created_at = utc_now_iso()
        claim_ids = [row["id"] for row in claim_rows if isinstance(row.get("id"), int)]
        if claim_ids:
            placeholders = ", ".join("?" for _ in claim_ids)
            conn.execute(
                f"DELETE FROM claim_citation_links WHERE verified_claim_id IN ({placeholders})",
                claim_ids,
            )

        citation_by_chunk_id = {
            str(row["chunk_id"]): row
            for row in citation_rows
            if row.get("chunk_id")
        }
        inserted_links: list[dict[str, Any]] = []

        for claim_row, claim in zip(claim_rows, claims):
            verification = claim.get("verification") or {}
            link_specs = [
                (
                    "supporting",
                    verification.get("best_supporting_chunk") or verification.get("best_chunk"),
                    verification.get("best_supporting_score", verification.get("final_score")),
                ),
                (
                    "contradicting",
                    verification.get("best_contradicting_chunk"),
                    verification.get("best_contradiction_score"),
                ),
            ]

            for relationship, chunk, score in link_specs:
                if not isinstance(chunk, dict):
                    continue
                chunk_id = chunk.get("id") or chunk.get("chunk_id")
                if not chunk_id:
                    continue
                citation_row = citation_by_chunk_id.get(str(chunk_id))
                if citation_row is None:
                    continue

                cursor = conn.execute(
                    """
                    INSERT INTO claim_citation_links (
                        verified_claim_id,
                        interaction_citation_id,
                        relationship,
                        score,
                        created_at,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        claim_row["id"],
                        citation_row["id"],
                        relationship,
                        score,
                        created_at,
                        json.dumps(
                            {
                                "claim_id": claim.get("claim_id"),
                                "chunk_id": chunk_id,
                                "relationship": relationship,
                                "verdict": verification.get("verdict"),
                            }
                        ),
                    ),
                )
                row = conn.execute(
                    """
                    SELECT
                        id,
                        verified_claim_id,
                        interaction_citation_id,
                        relationship,
                        score,
                        created_at,
                        metadata_json
                    FROM claim_citation_links
                    WHERE id = ?
                    """,
                    (cursor.lastrowid,),
                ).fetchone()
                inserted_links.append(dict(row))
        return inserted_links

    def list_verified_claims(self, interaction_id: int, user_id: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    vc.id,
                    vc.interaction_id,
                    vc.claim_text,
                    vc.verdict,
                    vc.confidence,
                    vc.source_citation,
                    vc.created_at,
                    vc.claim_id,
                    vc.claim_type,
                    vc.claim_source,
                    vc.certainty,
                    vc.doc_section,
                    vc.span_json,
                    vc.metadata_json
                FROM verified_claims vc
                JOIN interactions i ON i.id = vc.interaction_id
                JOIN conversations c ON c.id = i.conversation_id
                WHERE vc.interaction_id = ? AND c.user_id = ?
                ORDER BY vc.id ASC
                """,
                (interaction_id, user_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_contradictions(self, interaction_id: int, user_id: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    ct.id,
                    ct.interaction_id,
                    ct.chunk_i_id,
                    ct.chunk_j_id,
                    ct.score,
                    ct.created_at,
                    ct.metadata_json
                FROM contradictions ct
                JOIN interactions i ON i.id = ct.interaction_id
                JOIN conversations c ON c.id = i.conversation_id
                WHERE ct.interaction_id = ? AND c.user_id = ?
                ORDER BY ct.id ASC
                """,
                (interaction_id, user_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_interaction_citations(self, interaction_id: int, user_id: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    ic.id,
                    ic.interaction_id,
                    ic.doc_id,
                    ic.chunk_id,
                    ic.source_label,
                    ic.score,
                    ic.used_in_prompt,
                    ic.created_at,
                    ic.metadata_json
                FROM interaction_citations ic
                JOIN interactions i ON i.id = ic.interaction_id
                JOIN conversations c ON c.id = i.conversation_id
                WHERE ic.interaction_id = ? AND c.user_id = ?
                ORDER BY ic.id ASC
                """,
                (interaction_id, user_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_claim_citation_links(self, interaction_id: int, user_id: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    ccl.id,
                    ccl.verified_claim_id,
                    ccl.interaction_citation_id,
                    ccl.relationship,
                    ccl.score,
                    ccl.created_at,
                    ccl.metadata_json,
                    vc.claim_id,
                    ic.chunk_id,
                    ic.doc_id,
                    ic.source_label,
                    ic.metadata_json AS citation_metadata_json
                FROM claim_citation_links ccl
                JOIN verified_claims vc ON vc.id = ccl.verified_claim_id
                JOIN interaction_citations ic ON ic.id = ccl.interaction_citation_id
                JOIN interactions i ON i.id = vc.interaction_id
                JOIN conversations c ON c.id = i.conversation_id
                WHERE vc.interaction_id = ? AND c.user_id = ?
                ORDER BY vc.id ASC, ccl.id ASC
                """,
                (interaction_id, user_id),
            ).fetchall()
        return [dict(row) for row in rows]

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

    def list_recent_messages(
        self,
        conversation_id: int,
        user_id: int,
        *,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []

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
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]
