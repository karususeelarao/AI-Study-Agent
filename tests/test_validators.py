"""Unit tests for utils/validators.py."""

from __future__ import annotations

import pytest

from utils.validators import (
    ValidationError,
    validate_pdf_filename,
    validate_question,
    validate_topic,
)


class TestValidateTopic:
    def test_valid_topic_passes(self):
        assert validate_topic("Python Decorators") == "Python Decorators"

    def test_strips_whitespace(self):
        assert validate_topic("  SQL Joins  ") == "SQL Joins"

    def test_empty_topic_raises(self):
        with pytest.raises(ValidationError):
            validate_topic("")

    def test_too_short_topic_raises(self):
        with pytest.raises(ValidationError):
            validate_topic("a")

    def test_none_topic_raises(self):
        with pytest.raises(ValidationError):
            validate_topic(None)

    def test_too_long_topic_raises(self):
        with pytest.raises(ValidationError):
            validate_topic("x" * 200)


class TestValidatePdfFilename:
    def test_valid_pdf_passes(self):
        validate_pdf_filename("resume.pdf")  # should not raise

    def test_non_pdf_raises(self):
        with pytest.raises(ValidationError):
            validate_pdf_filename("resume.docx")

    def test_empty_filename_raises(self):
        with pytest.raises(ValidationError):
            validate_pdf_filename("")


class TestValidateQuestion:
    def test_valid_question_passes(self):
        assert validate_question("What is a hash map?") == "What is a hash map?"

    def test_empty_question_raises(self):
        with pytest.raises(ValidationError):
            validate_question("   ")

    def test_overly_long_question_raises(self):
        with pytest.raises(ValidationError):
            validate_question("x" * 1500)
