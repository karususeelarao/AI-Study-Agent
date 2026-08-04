"""
utils/validators.py
==============================================================================
Shared input validation helpers used by the Streamlit UI layer before any
request reaches an agent or service — first line of defense against
garbage input (empty topics, oversized files, disallowed file types).
==============================================================================
"""

from __future__ import annotations

from config import settings

ALLOWED_PDF_EXTENSIONS = (".pdf",)
MIN_TOPIC_LENGTH = 2
MAX_TOPIC_LENGTH = 120


class ValidationError(Exception):
    """Raised when user-provided input fails validation."""


def validate_topic(topic: str) -> str:
    """
    Validate and normalize a user-entered study topic.

    Returns:
        The cleaned topic string.

    Raises:
        ValidationError: if the topic is empty, too short, or too long.
    """
    if topic is None:
        raise ValidationError("Topic cannot be empty.")
    cleaned = topic.strip()
    if len(cleaned) < MIN_TOPIC_LENGTH:
        raise ValidationError(f"Topic must be at least {MIN_TOPIC_LENGTH} characters.")
    if len(cleaned) > MAX_TOPIC_LENGTH:
        raise ValidationError(f"Topic must be under {MAX_TOPIC_LENGTH} characters.")
    return cleaned


def validate_pdf_filename(filename: str) -> None:
    """Raise ValidationError if the filename doesn't look like a PDF."""
    if not filename or not filename.lower().endswith(ALLOWED_PDF_EXTENSIONS):
        raise ValidationError("Only .pdf files are supported.")


def validate_file_size(file_bytes: bytes) -> None:
    """Raise ValidationError if the file exceeds the configured max size."""
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise ValidationError(
            f"File is {size_mb:.1f} MB; the limit is {settings.max_upload_size_mb} MB."
        )


def validate_question(question: str) -> str:
    """Validate a free-text question before sending it to an agent."""
    if not question or not question.strip():
        raise ValidationError("Question cannot be empty.")
    cleaned = question.strip()
    if len(cleaned) > 1000:
        raise ValidationError("Question is too long (max 1000 characters).")
    return cleaned
