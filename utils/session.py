"""
utils/session.py
==============================================================================
Helpers for managing a stable per-browser-session identifier, used to
scope chat history and notes in SQLite per user session.
==============================================================================
"""

from __future__ import annotations

import uuid


def new_session_id() -> str:
    """Generate a new unique session identifier."""
    return str(uuid.uuid4())
