"""
utils/pdf_export.py
==============================================================================
Generates a downloadable PDF file from study notes using fpdf2.
==============================================================================
"""

from __future__ import annotations

from datetime import datetime, timezone


from fpdf import FPDF
from pathlib import Path

from config import settings
from logger import get_logger

logger = get_logger(__name__)


class PDFExportError(Exception):
    """Raised when PDF export fails."""


class NotesPDF(FPDF):
    def __init__(self, title: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.doc_title = title

        font_path = Path(__file__).parent.parent / "fonts" / "DejaVuSans.ttf"

        self.add_font("DejaVu", "", str(font_path))
        self.add_font("DejaVu", "B", str(font_path))

        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()

    def header(self):
        self.set_font("DejaVu", "B", 14)
        self.cell(0, 10, self.doc_title, new_x="LMARGIN", new_y="NEXT", align="C")

        self.set_font("DejaVu", "", 9)
        self.set_text_color(120, 120, 120)
        self.cell(
            0,
            6,
            datetime.now(timezone.utc).strftime("Generated %Y-%m-%d %H:%M UTC"),
            new_x="LMARGIN",
            new_y="NEXT",
            align="C",
        )

        self.set_text_color(0, 0, 0)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_heading(self, text: str):
        self.set_font("DejaVu", "B", 12)
        self.set_fill_color(230, 230, 250)
        self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT", fill=True)
        self.ln(2)

    def body_text(self, text: str):
        self.set_font("DejaVu", "", 11)
        self.multi_cell(0, 6, text)
        self.ln(2)


def export_notes_to_pdf(
    topic: str,
    sections: dict[str, str],
    output_dir: str | Path | None = None,
) -> Path:
    """
    Export study notes to a PDF.

    Returns:
        Path to the generated PDF.
    """

    if not sections:
        raise PDFExportError("No content provided.")

    output_dir = Path(output_dir or settings.notes_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_topic = "".join(
        c if c.isalnum() or c in (" ", "-", "_") else "_"
        for c in topic
    ).strip()

    filename = (
        f"{safe_topic.replace(' ', '_')}_notes_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pdf"
    )

    output_path = output_dir / filename

    try:
        pdf = NotesPDF(f"Study Notes: {topic}")

        for heading, content in sections.items():
            pdf.section_heading(heading)
            pdf.body_text(content if content.strip() else "(No content)")

        pdf.output(str(output_path))

        logger.info("PDF exported: %s", output_path)

        return output_path

    except Exception as exc:
        logger.exception("PDF export failed")
        raise PDFExportError(f"Failed to generate PDF: {exc}") from exc