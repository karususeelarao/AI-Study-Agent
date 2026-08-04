"""
Unit tests for services/llm_service.py.

These tests NEVER call the real OpenAI API — the underlying ChatOpenAI
client is mocked, so the test suite runs offline and costs $0.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from openai import AuthenticationError, RateLimitError

from services.llm_service import LLMService, LLMServiceError


@pytest.fixture
def llm_service() -> LLMService:
    service = LLMService()
    service.client = MagicMock()  # replace the real ChatOpenAI client
    return service


class TestGenerate:
    def test_generate_returns_text(self, llm_service):
        llm_service.client.invoke.return_value = MagicMock(content="Paris is the capital of France.")
        result = llm_service.generate("system prompt", "What is the capital of France?")
        assert result == "Paris is the capital of France."

    def test_generate_wraps_authentication_error(self, llm_service, mocker):
        mocker.patch.object(
            llm_service,
            "_call_llm",
            side_effect=AuthenticationError("bad key", response=MagicMock(), body=None),
        )
        with pytest.raises(LLMServiceError, match="Authentication"):
            llm_service.generate("sys", "prompt")

    def test_generate_passes_history(self, llm_service):
        llm_service.client.invoke.return_value = MagicMock(content="follow-up answer")
        history = [{"role": "user", "content": "first question"}, {"role": "assistant", "content": "first answer"}]
        result = llm_service.generate("sys", "second question", history=history)
        assert result == "follow-up answer"
        # Ensure invoke was called with a messages list including history + new prompt.
        call_args = llm_service.client.invoke.call_args[0][0]
        assert len(call_args) == 4  # system + 2 history turns + new user prompt


class TestGenerateJson:
    def test_parses_clean_json(self, llm_service):
        llm_service.client.invoke.return_value = MagicMock(content='{"answer": "42"}')
        result = llm_service.generate_json("sys", "prompt")
        assert result == {"answer": "42"}

    def test_strips_markdown_fences(self, llm_service):
        llm_service.client.invoke.return_value = MagicMock(
            content='```json\n{"answer": "42"}\n```'
        )
        result = llm_service.generate_json("sys", "prompt")
        assert result == {"answer": "42"}

    def test_invalid_json_raises_llm_service_error(self, llm_service):
        llm_service.client.invoke.return_value = MagicMock(content="this is not json at all")
        with pytest.raises(LLMServiceError, match="could not be parsed"):
            llm_service.generate_json("sys", "prompt")
