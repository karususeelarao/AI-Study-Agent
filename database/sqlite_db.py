"""
database/sqlite_db.py
==============================================================================
SQLite persistence layer for the AI Personal Study & Interview Assistant.

Responsibilities:
    - Store chat/conversation history (for memory + audit trail).
    - Store generated study notes (topic, level, content, type).
    - Provide a clean OOP repository-style interface so the rest of the
      app never writes raw SQL outside this file.

Design pattern: Repository pattern. `SQLiteDatabase` encapsulates all
persistence logic; callers never see SQL.
==============================================================================
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from config import settings
from logger import get_logger

logger = get_logger(__name__)


class DatabaseError(Exception):
    """Raised when a database operation fails."""


@dataclass
class ChatMessage:
    """Represents a single chat turn stored in the database."""

    id: Optional[int]
    session_id: str
    role: str  # "user" or "assistant"
    content: str
    created_at: str


@dataclass
class StudyNote:
    """Represents a saved study note/artifact."""

    id: Optional[int]
    session_id: str
    topic: str
    note_type: str  # explanation | interview_qa | quiz | flashcard | example | exercise
    level: str  # beginner | intermediate | advanced | n/a
    content: str
    created_at: str


class SQLiteDatabase:
    """
    Thread-safe-ish wrapper around SQLite for chat history and notes.

    Note: SQLite connections are not safe to share across threads, so this
    class opens a short-lived connection per operation via a context
    manager rather than holding one connection open for the app's
    lifetime. This is the recommended pattern for Streamlit apps, which
    can run callbacks on different threads.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = str(db_path or settings.sqlite_db_path)
        self._initialize_schema()
        logger.info("SQLiteDatabase initialized at %s", self.db_path)

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a SQLite connection with row factory and FK enforcement enabled."""
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            logger.exception("SQLite operation failed, rolled back.")
            raise DatabaseError(str(exc)) from exc
        finally:
            conn.close()

    def _initialize_schema(self) -> None:
        """Create tables if they do not already exist."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS study_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    note_type TEXT NOT NULL,
                    level TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chat_session
                    ON chat_messages (session_id);

                CREATE INDEX IF NOT EXISTS idx_notes_session
                    ON study_notes (session_id);
                """
            )
        logger.debug("Database schema verified/created.")

    # ------------------------------------------------------------------
    # Chat history
    # ------------------------------------------------------------------
    def add_chat_message(self, session_id: str, role: str, content: str) -> int:
        """Insert a chat message and return its new row id."""
        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid role: {role!r}. Must be 'user' or 'assistant'.")
        created_at = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO chat_messages (session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, role, content, created_at),
            )
            message_id = cursor.lastrowid
        logger.debug("Stored chat message id=%s session=%s role=%s", message_id, session_id, role)
        return message_id

    def get_chat_history(self, session_id: str, limit: int = 50) -> list[ChatMessage]:
        """Return chat history for a session, oldest first, capped at `limit`."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM ("
                "  SELECT * FROM chat_messages WHERE session_id = ? "
                "  ORDER BY id DESC LIMIT ?"
                ") sub ORDER BY id ASC",
                (session_id, limit),
            ).fetchall()
        return [
            ChatMessage(
                id=row["id"],
                session_id=row["session_id"],
                role=row["role"],
                content=row["content"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def clear_chat_history(self, session_id: str) -> None:
        """Delete all chat messages for a given session."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        logger.info("Cleared chat history for session=%s", session_id)

    # ------------------------------------------------------------------
    # Study notes
    # ------------------------------------------------------------------
    def save_note(
        self,
        session_id: str,
        topic: str,
        note_type: str,
        level: str,
        content: str,
    ) -> int:
        """Persist a generated study artifact (explanation, quiz, etc.)."""

        created_at = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO study_notes
                (session_id, topic, note_type, level, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    topic,
                    note_type,
                    level,
                    content,
                    created_at,
                ),
            )
            note_id = cursor.lastrowid

        logger.info(
            "Saved note id=%s topic=%s type=%s level=%s",
            note_id,
            topic,
            note_type,
            level,
        )

        return note_id

    def save_mock_interview(
        self,
        session_id: str,
        topic: str,
        question: str,
        answer: str,
        score: str,
    ) -> None:
        """Save a mock interview session."""

        created_at = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO study_notes
                (session_id, topic, note_type, level, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    topic,
                    "mock_interview",
                    score,
                    f"Question:\n{question}\n\nAnswer:\n{answer}",
                    created_at,
                ),
            )

        logger.info("Saved mock interview for topic=%s", topic)

    def get_notes(self, session_id: str, topic: Optional[str] = None) -> list[StudyNote]:
        """Fetch saved notes for a session, optionally filtered by topic."""
        with self._get_connection() as conn:
            if topic:
                rows = conn.execute(
                    "SELECT * FROM study_notes WHERE session_id = ? AND topic = ? "
                    "ORDER BY id DESC",
                    (session_id, topic),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM study_notes WHERE session_id = ? ORDER BY id DESC",
                    (session_id,),
                ).fetchall()
        return [
            StudyNote(
                id=row["id"],
                session_id=row["session_id"],
                topic=row["topic"],
                note_type=row["note_type"],
                level=row["level"],
                content=row["content"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def get_all_topics(self, session_id: str) -> list[str]:
        """Return distinct topics studied in this session."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT topic FROM study_notes WHERE session_id = ? ORDER BY topic",
                (session_id,),
            ).fetchall()
        return [row["topic"] for row in rows]

    def delete_note(self, note_id: int) -> None:
        """Delete a single note by id."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM study_notes WHERE id = ?", (note_id,))
        logger.info("Deleted note id=%s", note_id)
