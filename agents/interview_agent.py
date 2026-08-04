"""
agents/interview_agent.py
==============================================================================
InterviewAgent: generates realistic technical interview questions and
model answers for a given topic and difficulty level.
==============================================================================
"""

from __future__ import annotations

from logger import get_logger
from models.schemas import DifficultyLevel, InterviewQuestion
from services.llm_service import LLMService, LLMServiceError

logger = get_logger(__name__)


class InterviewAgentError(Exception):
    """Raised when the interview agent fails to produce valid content."""


class InterviewAgent:
    """Generates interview-style question/answer pairs."""

    SYSTEM_PROMPT = (
        "You are a senior technical interviewer at a top tech company. You "
        "write realistic, commonly-asked interview questions and concise, "
        "accurate model answers that a strong candidate would give."
    )

    def __init__(self, llm_service: LLMService) -> None:
        self.llm = llm_service

    def generate_questions(
        self, topic: str, level: DifficultyLevel, count: int = 5
    ) -> list[InterviewQuestion]:
        """Generate `count` interview Q&A pairs for a topic at a given level."""
        user_prompt = (
            f"Topic: {topic}\n"
            f"Difficulty: {level.value}\n\n"
            f"Generate exactly {count} realistic technical interview questions "
            f"about this topic at {level.value} level, each with a concise, "
            f"correct model answer (3-6 sentences).\n\n"
            "Return a JSON object with this exact shape:\n"
            "{\n"
            '  "questions": [\n'
            '    {"question": "...", "answer": "...", "difficulty": "' + level.value + '"}\n'
            "  ]\n"
            "}"
        )
        try:
            data = self.llm.generate_json(self.SYSTEM_PROMPT, user_prompt)
            raw_questions = data.get("questions", [])
            questions = []
            for item in raw_questions:
                item.setdefault("difficulty", level.value)
                questions.append(InterviewQuestion(**item))
            if not questions:
                raise InterviewAgentError("LLM returned zero interview questions.")
            return questions
        except LLMServiceError:
            raise
        except InterviewAgentError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("InterviewAgent failed for topic=%s level=%s", topic, level)
            raise InterviewAgentError(f"Failed to generate interview questions: {exc}") from exc

    def generate_all_levels(self, topic: str, count_per_level: int = 3) -> list[InterviewQuestion]:
        """Generate interview questions spanning all three difficulty levels."""
        all_questions: list[InterviewQuestion] = []
        for level in DifficultyLevel:
            all_questions.extend(self.generate_questions(topic, level, count_per_level))
        return all_questions
