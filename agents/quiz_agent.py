"""
agents/quiz_agent.py
==============================================================================
QuizAgent: generates multiple-choice quiz questions and flashcards for a
given topic and difficulty level.
==============================================================================
"""

from __future__ import annotations

from logger import get_logger
from models.schemas import DifficultyLevel, Flashcard, QuizQuestion
from services.llm_service import LLMService, LLMServiceError

logger = get_logger(__name__)


class QuizAgentError(Exception):
    """Raised when the quiz agent fails to produce valid content."""


class QuizAgent:
    """Generates quiz questions (MCQ) and flashcards."""

    SYSTEM_PROMPT = (
        "You are an expert exam-question writer for technical certification "
        "and interview-prep courses. You write unambiguous multiple-choice "
        "questions with exactly one correct answer, and concise flashcards "
        "that test recall of key facts."
    )

    def __init__(self, llm_service: LLMService) -> None:
        self.llm = llm_service

    def generate_quiz(
        self, topic: str, level: DifficultyLevel, count: int = 5
    ) -> list[QuizQuestion]:
        """Generate `count` multiple-choice quiz questions."""
        user_prompt = (
            f"Topic: {topic}\n"
            f"Difficulty: {level.value}\n\n"
            f"Generate exactly {count} multiple-choice quiz questions about this "
            "topic. Each question must have exactly 4 options with only one "
            "correct answer, plus a one-sentence explanation of why the correct "
            "answer is correct.\n\n"
            "Return a JSON object with this exact shape:\n"
            "{\n"
            '  "quiz": [\n'
            "    {\n"
            '      "question": "...",\n'
            '      "options": ["...", "...", "...", "..."],\n'
            '      "correct_answer_index": 0,\n'
            '      "explanation": "..."\n'
            "    }\n"
            "  ]\n"
            "}"
        )
        try:
            data = self.llm.generate_json(self.SYSTEM_PROMPT, user_prompt)
            raw_quiz = data.get("quiz", [])
            questions = [QuizQuestion(**item) for item in raw_quiz]
            if not questions:
                raise QuizAgentError("LLM returned zero quiz questions.")
            return questions
        except LLMServiceError:
            raise
        except QuizAgentError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("QuizAgent failed to generate quiz for topic=%s", topic)
            raise QuizAgentError(f"Failed to generate quiz questions: {exc}") from exc

    def generate_flashcards(self, topic: str, count: int = 8) -> list[Flashcard]:
        """Generate front/back flashcards for a topic."""
        user_prompt = (
            f"Topic: {topic}\n\n"
            f"Generate exactly {count} flashcards for this topic. Each flashcard's "
            '"front" should be a short question or term, and "back" should be a '
            "concise, accurate answer/definition (1-3 sentences).\n\n"
            "Return a JSON object with this exact shape:\n"
            "{\n"
            '  "flashcards": [\n'
            '    {"front": "...", "back": "..."}\n'
            "  ]\n"
            "}"
        )
        try:
            data = self.llm.generate_json(self.SYSTEM_PROMPT, user_prompt)
            raw_cards = data.get("flashcards", [])
            cards = [Flashcard(**item) for item in raw_cards]
            if not cards:
                raise QuizAgentError("LLM returned zero flashcards.")
            return cards
        except LLMServiceError:
            raise
        except QuizAgentError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("QuizAgent failed to generate flashcards for topic=%s", topic)
            raise QuizAgentError(f"Failed to generate flashcards: {exc}") from exc
