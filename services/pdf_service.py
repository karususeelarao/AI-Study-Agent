"""
services/pdf_service.py
==============================================================================
Handles reading uploaded PDF files and splitting their text into chunks
suitable for embedding (RAG ingestion pipeline).
==============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from config import settings
from logger import get_logger

logger = get_logger(__name__)


class PDFServiceError(Exception):
    """Raised when PDF reading or processing fails."""


@dataclass
class PDFChunk:
    """A single chunk of text extracted from a PDF, with page metadata."""

    text: str
    page_number: int
    source_filename: str


class PDFService:
    """Reads PDFs from disk/bytes and splits them into embeddable chunks."""

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        logger.info(
            "PDFService ready (chunk_size=%d, chunk_overlap=%d)",
            self.chunk_size,
            self.chunk_overlap,
        )

    def validate_file_size(self, file_bytes: bytes, filename: str) -> None:
        """Raise PDFServiceError if the uploaded file exceeds the configured limit."""
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > settings.max_upload_size_mb:
            raise PDFServiceError(
                f"'{filename}' is {size_mb:.1f} MB, which exceeds the "
                f"{settings.max_upload_size_mb} MB upload limit."
            )

    def save_upload(self, file_bytes: bytes, filename: str) -> Path:
        """Persist an uploaded file's bytes to the uploads directory."""
        self.validate_file_size(file_bytes, filename)
        destination = settings.upload_dir / filename
        try:
            destination.write_bytes(file_bytes)
            logger.info("Saved upload '%s' (%d bytes)", filename, len(file_bytes))
            return destination
        except OSError as exc:
            logger.exception("Failed to save uploaded file '%s'", filename)
            raise PDFServiceError(f"Could not save uploaded file: {exc}") from exc

    def extract_text_by_page(self, pdf_path: Path) -> list[tuple[int, str]]:
        """
        Extract text from a PDF, page by page.

        Returns:
            List of (page_number, text) tuples for pages containing text.
            Pages with no extractable text (e.g. scanned images) are skipped
            with a warning, rather than crashing the whole pipeline.
        """
        if not pdf_path.exists():
            raise PDFServiceError(f"PDF file not found: {pdf_path}")

        try:
            reader = PdfReader(str(pdf_path))
        except PdfReadError as exc:
            logger.exception("Corrupt or unreadable PDF: %s", pdf_path)
            raise PDFServiceError(f"Could not read PDF '{pdf_path.name}': {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error opening PDF: %s", pdf_path)
            raise PDFServiceError(f"Unexpected error opening PDF '{pdf_path.name}': {exc}") from exc

        if reader.is_encrypted:
            raise PDFServiceError(
                f"'{pdf_path.name}' is password-protected. Please upload an unlocked PDF."
            )

        pages: list[tuple[int, str]] = []
        for page_index, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:  # noqa: BLE001
                logger.warning("Failed to extract text from page %d of %s", page_index, pdf_path.name)
                continue
            if text.strip():
                pages.append((page_index, text))
            else:
                logger.warning(
                    "Page %d of '%s' has no extractable text (likely scanned image).",
                    page_index,
                    pdf_path.name,
                )

        if not pages:
            raise PDFServiceError(
                f"No extractable text found in '{pdf_path.name}'. "
                f"It may be a scanned/image-only PDF (OCR not supported)."
            )

        logger.info("Extracted text from %d/%d pages of '%s'", len(pages), len(reader.pages), pdf_path.name)
        return pages

    def chunk_pdf(self, pdf_path: Path) -> list[PDFChunk]:
        """
        Full pipeline: extract text per page, then split into overlapping
        chunks suitable for embedding.
        """
        pages = self.extract_text_by_page(pdf_path)
        chunks: list[PDFChunk] = []
        for page_number, page_text in pages:
            for split_text in self.splitter.split_text(page_text):
                if split_text.strip():
                    chunks.append(
                        PDFChunk(
                            text=split_text.strip(),
                            page_number=page_number,
                            source_filename=pdf_path.name,
                        )
                    )
        logger.info("Split '%s' into %d chunks", pdf_path.name, len(chunks))
        return chunks
