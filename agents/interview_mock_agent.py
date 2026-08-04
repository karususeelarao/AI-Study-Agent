"""
agents/interview_mock_agent.py
=========================================================
Mock Interview Agent
=========================================================
"""

from __future__ import annotations

import json

from logger import get_logger
from services.llm_service import LLMService, LLMServiceError

logger = get_logger(__name__)


class MockInterviewAgentError(Exception):
    pass


class MockInterviewAgent:

    SYSTEM_PROMPT = (
        "You are an experienced software engineer and technical interviewer. "
        "Conduct realistic technical interviews. "
        "Ask only ONE interview question at a time. "
        "After receiving the candidate's answer, evaluate it professionally."
    )

    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    def generate_question(
        self,
        topic: str,
        difficulty: str,
    ) -> str:

        prompt = f"""
Topic: {topic}

Difficulty: {difficulty}

Generate ONE technical interview question.

Return ONLY the question.
"""

        return self.llm.generate(
            self.SYSTEM_PROMPT,
            prompt,
        )

    def evaluate_answer(
        self,
        topic: str,
        question: str,
        answer: str,
    ) -> dict:

        prompt = f"""
Topic:
{topic}

Question:
{question}

Candidate Answer:
{answer}

Return ONLY valid JSON.

{{
    "score": 0-10,
    "strengths": [
        "...",
        "..."
    ],
    "weaknesses": [
        "...",
        "..."
    ],
    "ideal_answer": "..."
}}
"""

        try:
            return self.llm.generate_json(
                self.SYSTEM_PROMPT,
                prompt,
            )

        except LLMServiceError as exc:
            logger.exception("Interview evaluation failed")
            raise MockInterviewAgentError(str(exc)) from exc