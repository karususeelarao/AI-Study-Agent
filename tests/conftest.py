"""
tests/conftest.py
==============================================================================
Shared pytest fixtures. Sets required env vars BEFORE any app module is
imported, so config.py's fail-fast validation passes during tests.
==============================================================================
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# --- Ensure test env vars exist before `config.py` is imported anywhere ---
os.environ.setdefault("GEMINI_API_KEY", "sk-test-fake-key-for-unit-tests")
os.environ.setdefault("LOG_LEVEL", "WARNING")

# Make the project root importable when running `pytest` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Provide a throwaway SQLite file path for isolated DB tests."""
    return tmp_path / "test_study_assistant.db"


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """
    Provide minimal valid PDF bytes (a blank single-page PDF) for testing
    the PDF service without needing a real file on disk.
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, "This is a test PDF about Python decorators.")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        pdf.output(tmp.name)
        return Path(tmp.name).read_bytes()
