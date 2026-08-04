"""Unit tests for services/pdf_service.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.pdf_service import PDFService, PDFServiceError


@pytest.fixture
def pdf_service() -> PDFService:
    return PDFService(chunk_size=200, chunk_overlap=20)


class TestPDFService:
    def test_save_and_extract_text(self, pdf_service, sample_pdf_bytes, tmp_path):
        # Write directly to tmp_path rather than monkeypatching settings
        # (Settings is an immutable/frozen dataclass by design — see config.py).
        saved_path = tmp_path / "test.pdf"
        saved_path.write_bytes(sample_pdf_bytes)
        assert saved_path.exists()

        pages = pdf_service.extract_text_by_page(saved_path)
        assert len(pages) >= 1
        assert "decorators" in pages[0][1].lower() or "test" in pages[0][1].lower()

    def test_missing_file_raises(self, pdf_service):
        with pytest.raises(PDFServiceError, match="not found"):
            pdf_service.extract_text_by_page(Path("/nonexistent/file.pdf"))

    def test_file_size_validation(self, pdf_service):
        huge_bytes = b"0" * (25 * 1024 * 1024)  # 25 MB > default 20 MB limit
        with pytest.raises(PDFServiceError, match="exceeds"):
            pdf_service.validate_file_size(huge_bytes, "big.pdf")

    def test_chunk_pdf_produces_chunks(self, pdf_service, sample_pdf_bytes, tmp_path):
        saved_path = tmp_path / "test2.pdf"
        saved_path.write_bytes(sample_pdf_bytes)
        chunks = pdf_service.chunk_pdf(saved_path)
        assert len(chunks) >= 1
        assert all(chunk.source_filename == "test2.pdf" for chunk in chunks)
