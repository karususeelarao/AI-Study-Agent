"""Unit tests for config.py — validation and defaults."""

from __future__ import annotations

import pytest

from config import ConfigError, Settings


class TestSettingsValidation:
    def test_valid_settings_pass(self):
        s = Settings(gemini_api_key="sk-fake", chunk_size=1000, chunk_overlap=150, gemini_temperature=0.4)
        s.validate()  # should not raise

    def test_missing_api_key_raises(self):
        s = Settings(gemini_api_key="")
        with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
            s.validate()

    def test_overlap_greater_than_chunk_size_raises(self):
        s = Settings(gemini_api_key="sk-fake", chunk_size=100, chunk_overlap=200)
        with pytest.raises(ConfigError, match="CHUNK_OVERLAP"):
            s.validate()

    def test_temperature_out_of_range_raises(self):
        s = Settings(gemini_api_key="sk-fake", gemini_temperature=3.5)
        with pytest.raises(ConfigError, match="GEMINI_TEMPERATURE"):
            s.validate()

    def test_negative_top_k_raises(self):
        s = Settings(gemini_api_key="sk-fake", retrieval_top_k=0)
        with pytest.raises(ConfigError, match="RETRIEVAL_TOP_K"):
            s.validate()

    def test_settings_are_immutable(self):
        s = Settings(gemini_api_key="sk-fake")
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            s.gemini_api_key = "changed"
