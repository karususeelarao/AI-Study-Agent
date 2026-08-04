"""
agents/study_agent.py
==============================================================================
StudyAgent: generates beginner/intermediate/advanced explanations of a
topic, plus real-world examples and coding exercises.

Design pattern: each Agent class owns its own prompt templates and knows
how to turn raw LLM text/JSON into validated Pydantic models. Agents
depend on LLMService (injected) rather than constructing it themselves —
this is constructor-based dependency injection, which makes agents easy
to unit test with a mocked LLMService.
==============================================================================
"""

from __future__ import annotations

from logger import get_logger
from models.schemas import CodingExercise, DifficultyLevel, TopicExplanation
from services.llm_service import LLMService, LLMServiceError

logger = get_logger(__name__)


class StudyAgentError(Exception):
    """Raised when the study agent fails to produce valid content."""


class StudyAgent:
    """Generates core educational content for a given topic."""

    SYSTEM_PROMPT = (
        "You are an expert technical educator and mentor who explains "
        "programming, computer science, and technology topics with clarity "
        "and precision. You adapt your explanations to the learner's level "
        "without ever being condescending or inaccurate."
    )

    def __init__(self, llm_service: LLMService) -> None:
        self.llm = llm_service

    def explain_topic(self, topic: str, level: DifficultyLevel) -> TopicExplanation:
        """Generate a single-level explanation of a topic."""
        level_guidance = {
            DifficultyLevel.BEGINNER: (
                "Explain as if to someone new to programming. Avoid jargon, "
                "use simple analogies, and define any technical term you use."
            ),
            DifficultyLevel.INTERMEDIATE: (
                "Explain to someone with working programming experience. "
                "Cover how it works under the hood and common use cases."
            ),
            DifficultyLevel.ADVANCED: (
                "Explain to an experienced engineer preparing for a senior "
                "technical interview. Cover internals, edge cases, performance "
                "tradeoffs, and how it compares to alternative approaches."
            ),
        }
        user_prompt = (
            f"Topic: {topic}\n\n"
            f"Level: {level.value}\n"
            f"Guidance: {level_guidance[level]}\n\n"
            "Write a clear, well-structured explanation (250-400 words). "
            "Use short paragraphs and, where useful, a brief bullet list."
        )
        try:
            content = self.llm.generate(self.SYSTEM_PROMPT, user_prompt)
            return TopicExplanation(level=level, content=content)
        except LLMServiceError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("StudyAgent failed to explain topic=%s level=%s", topic, level)
            raise StudyAgentError(f"Failed to generate {level.value} explanation: {exc}") from exc

    def explain_all_levels(self, topic: str) -> list[TopicExplanation]:
        """Generate explanations at all three difficulty levels."""
        return [self.explain_topic(topic, level) for level in DifficultyLevel]

    def generate_real_world_examples(self, topic: str, count: int = 3) -> list[str]:
        """Generate real-world application examples for a topic."""
        user_prompt = (
            f"Topic: {topic}\n\n"
            f"List exactly {count} concrete real-world examples of where/how this "
            f"is used in industry (companies, products, or systems). "
            f"Return them as a numbered list, one to two sentences each."
        )
        try:
            raw = self.llm.generate(self.SYSTEM_PROMPT, user_prompt)
            examples = self._parse_numbered_list(raw)
            return examples[:count] if examples else [raw.strip()]
        except LLMServiceError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("StudyAgent failed to generate examples for topic=%s", topic)
            raise StudyAgentError(f"Failed to generate real-world examples: {exc}") from exc

    def generate_coding_exercises(
        self, topic: str, level: DifficultyLevel, count: int = 2
    ) -> list[CodingExercise]:
        """Generate coding practice exercises for a topic, as structured JSON."""
        user_prompt = (
            f"Topic: {topic}\n"
            f"Difficulty: {level.value}\n\n"
            f"Generate exactly {count} coding exercises for practicing this topic. "
            "Return a JSON object with this exact shape:\n"
            "{\n"
            '  "exercises": [\n'
            "    {\n"
            '      "title": "short exercise title",\n'
            '      "problem_statement": "clear description of the task",\n'
            '      "starter_code": "minimal starter code or function signature",\n'
            '      "solution": "a correct, idiomatic solution",\n'
            '      "hints": ["hint 1", "hint 2"]\n'
            "    }\n"
            "  ]\n"
            "}"
        )
        try:
            data = self.llm.generate_json(self.SYSTEM_PROMPT, user_prompt)
            raw_exercises = data.get("exercises", [])
            return [CodingExercise(**ex) for ex in raw_exercises]
        except LLMServiceError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("StudyAgent failed to generate coding exercises for topic=%s", topic)
            raise StudyAgentError(f"Failed to generate coding exercises: {exc}") from exc

    @staticmethod
    def _parse_numbered_list(raw_text: str) -> list[str]:
        """Parse a numbered list ('1. ...', '2. ...') into clean strings."""
        items: list[str] = []
        for line in raw_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # Strip leading "1.", "1)", "-", "*" style markers.
            for marker_len in range(1, 4):
                prefix = stripped[:marker_len]
                if prefix.rstrip(".)").isdigit():
                    stripped = stripped[marker_len:].lstrip(".) ").strip()
                    break
            if stripped.startswith(("-", "*")):
                stripped = stripped[1:].strip()
            if stripped:
                items.append(stripped)
        return items
