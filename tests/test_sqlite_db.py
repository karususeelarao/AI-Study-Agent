"""Unit tests for database/sqlite_db.py — the repository layer."""

from __future__ import annotations

import pytest

from database.sqlite_db import SQLiteDatabase


@pytest.fixture
def db(temp_db_path):
    return SQLiteDatabase(db_path=temp_db_path)


class TestChatHistory:
    def test_add_and_get_chat_message(self, db):
        db.add_chat_message("session-1", "user", "What is a decorator?")
        db.add_chat_message("session-1", "assistant", "A decorator wraps a function...")
        history = db.get_chat_history("session-1")
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"

    def test_chat_history_scoped_by_session(self, db):
        db.add_chat_message("session-A", "user", "Hello A")
        db.add_chat_message("session-B", "user", "Hello B")
        assert len(db.get_chat_history("session-A")) == 1
        assert len(db.get_chat_history("session-B")) == 1

    def test_invalid_role_raises(self, db):
        with pytest.raises(ValueError):
            db.add_chat_message("session-1", "system", "not allowed")

    def test_clear_chat_history(self, db):
        db.add_chat_message("session-1", "user", "test")
        db.clear_chat_history("session-1")
        assert db.get_chat_history("session-1") == []

    def test_history_respects_limit(self, db):
        for i in range(10):
            db.add_chat_message("session-1", "user", f"msg {i}")
        history = db.get_chat_history("session-1", limit=3)
        assert len(history) == 3
        # Should be the 3 most recent, in chronological order.
        assert history[-1].content == "msg 9"


class TestStudyNotes:
    def test_save_and_get_notes(self, db):
        db.save_note("session-1", "Python", "explanation", "beginner", "Python is a language...")
        notes = db.get_notes("session-1", topic="Python")
        assert len(notes) == 1
        assert notes[0].topic == "Python"
        assert notes[0].note_type == "explanation"

    def test_get_all_topics_is_distinct(self, db):
        db.save_note("session-1", "Python", "explanation", "beginner", "...")
        db.save_note("session-1", "Python", "quiz", "beginner", "...")
        db.save_note("session-1", "SQL", "explanation", "beginner", "...")
        topics = db.get_all_topics("session-1")
        assert sorted(topics) == ["Python", "SQL"]

    def test_delete_note(self, db):
        note_id = db.save_note("session-1", "Python", "flashcard", "n/a", "front->back")
        db.delete_note(note_id)
        assert db.get_notes("session-1", topic="Python") == []
