"""
Unit tests for the agent layer, using a mocked LLMService so no real API
calls are made. This demonstrates dependency injection making agents
trivially testable in isolation.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agents.interview_agent import InterviewAgent
from agents.quiz_agent import QuizAgent
from agents.study_agent import StudyAgent
from models.schemas import DifficultyLevel


@pytest.fixture
def mock_llm():
    return MagicMock()


class TestStudyAgent:
    def test_explain_topic_returns_explanation(self, mock_llm):
        mock_llm.generate.return_value = "Python is a high-level language."
        agent = StudyAgent(mock_llm)
        result = agent.explain_topic("Python", DifficultyLevel.BEGINNER)
        assert result.level == DifficultyLevel.BEGINNER
        assert "Python" in result.content

    def test_explain_all_levels_calls_llm_three_times(self, mock_llm):
        mock_llm.generate.return_value = "some explanation"
        agent = StudyAgent(mock_llm)
        results = agent.explain_all_levels("SQL")
        assert len(results) == 3
        assert mock_llm.generate.call_count == 3

    def test_parse_numbered_list(self):
        raw = "1. First example\n2. Second example\n3. Third example"
        parsed = StudyAgent._parse_numbered_list(raw)
        assert parsed == ["First example", "Second example", "Third example"]


class TestInterviewAgent:
    def test_generate_questions_parses_json(self, mock_llm):
        mock_llm.generate_json.return_value = {
            "questions": [
                {"question": "What is a list?", "answer": "An ordered collection.", "difficulty": "beginner"}
            ]
        }
        agent = InterviewAgent(mock_llm)
        questions = agent.generate_questions("Python", DifficultyLevel.BEGINNER, count=1)
        assert len(questions) == 1
        assert questions[0].question == "What is a list?"


class TestQuizAgent:
    def test_generate_quiz_parses_json(self, mock_llm):
        mock_llm.generate_json.return_value = {
            "quiz": [
                {
                    "question": "2+2=?",
                    "options": ["3", "4", "5", "6"],
                    "correct_answer_index": 1,
                    "explanation": "Basic math.",
                }
            ]
        }
        agent = QuizAgent(mock_llm)
        quiz = agent.generate_quiz("Math", DifficultyLevel.BEGINNER, count=1)
        assert len(quiz) == 1
        assert quiz[0].options[quiz[0].correct_answer_index] == "4"

    def test_generate_flashcards_parses_json(self, mock_llm):
        mock_llm.generate_json.return_value = {
            "flashcards": [{"front": "What is a decorator?", "back": "A function wrapper."}]
        }
        agent = QuizAgent(mock_llm)
        cards = agent.generate_flashcards("Python", count=1)
        assert len(cards) == 1
        assert cards[0].front == "What is a decorator?"
