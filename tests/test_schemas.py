"""Unit tests for models/schemas.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from models.schemas import QuizQuestion


class TestQuizQuestion:
    def test_valid_quiz_question(self):
        q = QuizQuestion(
            question="What is 2+2?",
            options=["3", "4", "5", "6"],
            correct_answer_index=1,
            explanation="Basic arithmetic.",
        )
        assert q.options[q.correct_answer_index] == "4"

    def test_out_of_range_index_raises(self):
        with pytest.raises(ValidationError):
            QuizQuestion(
                question="What is 2+2?",
                options=["3", "4"],
                correct_answer_index=5,
            )

    def test_too_few_options_raises(self):
        with pytest.raises(ValidationError):
            QuizQuestion(question="Q?", options=["only one"], correct_answer_index=0)
