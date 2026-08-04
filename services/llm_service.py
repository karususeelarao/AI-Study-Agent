"""
services/llm_service.py
==============================================================================
Async wrapper around the Gemini Chat Completions API (via LangChain's
ChatGoogleGenerativeAI), with retry/backoff and graceful error handling.

This is the ONLY module in the codebase that talks directly to the LLM.
Agents call `LLMService.generate()` / `LLMService.generate_json()` rather
than importing Gemini/LangChain themselves — this keeps the provider
swappable (e.g. to Anthropic or a local model) behind one interface.
==============================================================================
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings
from logger import get_logger
logger = get_logger(__name__)


class LLMServiceError(Exception):
    """Raised when the LLM fails to produce a usable response after retries."""


class LLMService:
    """
    Central gateway to the LLM. Wraps LangChain's ChatGoogleGenerativeAI with:
    - Exponential-backoff retries on transient errors.
    - Clear, user-safe error messages on permanent failures.
    - JSON-mode helper for structured generation.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.model = model or settings.gemini_model
        self.temperature = (
            temperature
            if temperature is not None
            else settings.gemini_temperature
        )
        self.max_tokens = max_tokens or settings.gemini_max_tokens

        self.client = ChatGoogleGenerativeAI(
        model=self.model,
        google_api_key=settings.gemini_api_key,
    temperature=self.temperature,
    max_output_tokens=self.max_tokens,
)

        logger.info(
            "LLMService ready (model=%s, temperature=%.2f)",
            self.model,
            self.temperature,
        )

    @staticmethod
    def _to_langchain_messages(
        system_prompt: str,
        history: list[dict[str, str]],
        user_prompt: str,
    ) -> list:
        """
        Convert system prompt + conversation history + user prompt
        into LangChain message objects.
        """
        messages = [SystemMessage(content=system_prompt)]

        for turn in history:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                messages.append(AIMessage(content=turn["content"]))

        messages.append(HumanMessage(content=user_prompt))
        return messages

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type(Exception),
    )
    def _call_llm(self, messages: list) -> str:
        response = self.client.invoke(messages)
        return response.text

    async def _call_llm_async(self, messages: list) -> str:
        response = await self.client.ainvoke(messages)
        return response.text
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Synchronous text generation.
        """
        messages = self._to_langchain_messages(
            system_prompt,
            history or [],
            user_prompt,
        )

        try:
            return self._call_llm(messages)

        

        except Exception as exc:
            import traceback

            traceback.print_exc()
            print("ACTUAL ERROR:", repr(exc))

            logger.exception("Unexpected error during LLM generation")

            raise LLMServiceError(
                f"Unexpected error while generating response: {exc}"
            ) from exc

    async def generate_async(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Async version of generate().
        """
        messages = self._to_langchain_messages(
            system_prompt,
            history or [],
            user_prompt,
        )

        try:
            return await self._call_llm_async(messages)

        
        except Exception as exc:
            logger.exception(
                "Unexpected error during async LLM generation"
            )
            raise LLMServiceError(
                f"Unexpected error while generating response: {exc}"
            ) from exc

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """
        Generate structured JSON output.
        """

        json_system_prompt = (
            f"{system_prompt}\n\n"
            "IMPORTANT: Respond with ONLY valid JSON. "
            "Do not include markdown or explanations."
        )

        raw = self.generate(
            json_system_prompt,
            user_prompt,
            history,
        )

        cleaned = raw.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]

        if cleaned.startswith("```"):
            cleaned = cleaned[3:]

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)

        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse LLM JSON output: %s | raw=%s",
                exc,
                raw[:500],
            )

            raise LLMServiceError(
                "The AI returned invalid JSON. Please try again."
            ) from exc