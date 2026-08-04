"""
agents/pdf_agent.py
==============================================================================
PDFAgent: implements Retrieval-Augmented Generation (RAG) over uploaded
PDFs. Retrieves relevant chunks from the FAISS vector store, then asks
the LLM to answer strictly grounded in that retrieved context.

Also owns simple conversation memory (a rolling window of recent turns)
so follow-up questions like "what about part 2?" resolve correctly.
==============================================================================
"""

from __future__ import annotations

from pathlib import Path

from logger import get_logger
from models.schemas import PDFQAResult
from services.llm_service import LLMService, LLMServiceError
from services.pdf_service import PDFService, PDFServiceError
from services.vector_service import VectorService, VectorServiceError

logger = get_logger(__name__)


class PDFAgentError(Exception):
    """Raised when the PDF RAG agent fails to process or answer."""


class PDFAgent:
    """Retrieval-augmented question answering over uploaded PDF documents."""

    SYSTEM_PROMPT = (
        "You are a helpful study assistant answering questions about a "
        "document the user uploaded. Answer ONLY using the provided context "
        "excerpts. If the answer is not contained in the context, say clearly "
        "that the document does not appear to cover that, rather than "
        "guessing or using outside knowledge."
    )

    MAX_MEMORY_TURNS = 6  # keep the last N turns (user+assistant pairs count as 2 each)

    def __init__(
        self,
        llm_service: LLMService,
        pdf_service: PDFService,
        vector_service: VectorService,
    ) -> None:
        self.llm = llm_service
        self.pdf_service = pdf_service
        self.vector_service = vector_service
        self._memory: list[dict[str, str]] = []

    def ingest_pdf(self, pdf_path: Path) -> int:
        """
        Process a PDF end-to-end: extract text, chunk it, embed it, and
        add it to the vector store.

        Returns:
            Number of chunks indexed.
        """
        try:
            chunks = self.pdf_service.chunk_pdf(pdf_path)
            texts = [c.text for c in chunks]
            metadatas = [
                {"source": c.source_filename, "page": c.page_number} for c in chunks
            ]
            count = self.vector_service.add_texts(texts, metadatas)
            self.vector_service.save()
            logger.info("Ingested '%s' -> %d chunks indexed", pdf_path.name, count)
            return count
        except (PDFServiceError, VectorServiceError):
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("PDFAgent failed to ingest '%s'", pdf_path)
            raise PDFAgentError(f"Failed to process PDF '{pdf_path.name}': {exc}") from exc

    def ask(self, question: str) -> PDFQAResult:
        """
        Answer a question using RAG: retrieve relevant chunks, then ask
        the LLM to answer grounded strictly in that context.
        """
        if not question or not question.strip():
            raise PDFAgentError("Question must not be empty.")

        try:
            retrieved = self.vector_service.similarity_search(question)
        except VectorServiceError as exc:
            raise PDFAgentError(f"Failed to retrieve context: {exc}") from exc

        if not retrieved:
            note = (
                "No document has been uploaded/indexed yet, so this answer is "
                "not grounded in any document."
            )
            answer = (
                "I don't have any uploaded document content to search yet. "
                "Please upload a PDF first."
            )
            self._remember("user", question)
            self._remember("assistant", answer)
            return PDFQAResult(question=question, answer=answer, source_chunks=[], confidence_note=note)

        context_block = "\n\n---\n\n".join(
            f"[Source: {chunk.metadata.get('source', 'unknown')}, "
            f"page {chunk.metadata.get('page', '?')}]\n{chunk.text}"
            for chunk in retrieved
        )
        user_prompt = (
            f"Context excerpts from the uploaded document:\n\n{context_block}\n\n"
            f"---\n\nQuestion: {question}\n\n"
            "Answer the question using only the context above. Cite the page "
            "number(s) you used where relevant."
        )

        try:
            answer = self.llm.generate(self.SYSTEM_PROMPT, user_prompt, history=self._memory)
        except LLMServiceError:
            raise

        self._remember("user", question)
        self._remember("assistant", answer)

        avg_score = sum(c.score for c in retrieved) / len(retrieved)
        confidence_note = None
        if avg_score < 0.3:
            confidence_note = (
                "Low similarity between the question and document content — "
                "the answer may not be well grounded."
            )

        return PDFQAResult(
            question=question,
            answer=answer,
            source_chunks=[c.text for c in retrieved],
            confidence_note=confidence_note,
        )

    def _remember(self, role: str, content: str) -> None:
        """Append a turn to the rolling memory window, trimming old turns."""
        self._memory.append({"role": role, "content": content})
        if len(self._memory) > self.MAX_MEMORY_TURNS:
            self._memory = self._memory[-self.MAX_MEMORY_TURNS :]

    def reset_memory(self) -> None:
        """Clear conversation memory (e.g. when a new PDF is uploaded)."""
        self._memory = []
        logger.info("PDFAgent conversation memory reset.")

    def get_memory(self) -> list[dict[str, str]]:
        """Return the current rolling memory window (for display/debugging)."""
        return list(self._memory)
