"""
services/embedding_service.py
==============================================================================
Wraps Sentence Transformers to produce local, free text embeddings.

Why local embeddings instead of OpenAI's embedding API:
    - Zero marginal cost per PDF chunk embedded (can be thousands of chunks).
    - No network round-trip -> faster for bulk indexing.
    - Keeps sensitive document content off the OpenAI API entirely if
      desired (privacy-friendly RAG).
==============================================================================
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings
from logger import get_logger

logger = get_logger(__name__)


class EmbeddingServiceError(Exception):
    """Raised when embedding generation fails."""


class EmbeddingService:
    """
    Thin OOP wrapper around a SentenceTransformer model.

    The underlying model is loaded once (expensive: loads weights into
    memory) and cached via `_load_model`, so repeated instantiation of
    this service (e.g. across Streamlit re-runs) does not reload weights.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model_name
        self.model = self._load_model(self.model_name)
        logger.info("EmbeddingService ready with model=%s", self.model_name)

    @staticmethod
    @lru_cache(maxsize=2)
    def _load_model(model_name: str) -> SentenceTransformer:
        """
        Load (and cache) a SentenceTransformer model by name.
        `lru_cache` ensures the same model object is reused across every
        EmbeddingService instance in the process.
        """
        logger.info("Loading embedding model '%s' (first load may take a moment)...", model_name)
        try:
            return SentenceTransformer(model_name)
        except Exception as exc:  # noqa: BLE001 - surface any load failure clearly
            logger.exception("Failed to load embedding model '%s'", model_name)
            raise EmbeddingServiceError(
                f"Could not load embedding model '{model_name}': {exc}"
            ) from exc

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a batch of texts.

        Args:
            texts: List of raw text chunks.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim), dtype float32.
        """
        if not texts:
            raise EmbeddingServiceError("Cannot embed an empty list of texts.")
        cleaned = [t.strip() for t in texts if t and t.strip()]
        if not cleaned:
            raise EmbeddingServiceError("All provided texts were empty after stripping.")
        try:
            embeddings = self.model.encode(
                cleaned,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # cosine similarity via inner product
            )
            return embeddings.astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Embedding generation failed for batch of %d texts", len(cleaned))
            raise EmbeddingServiceError(f"Failed to generate embeddings: {exc}") from exc

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string (used at retrieval time)."""
        if not query or not query.strip():
            raise EmbeddingServiceError("Cannot embed an empty query.")
        return self.embed_texts([query])[0]

    def get_embedding_dimension(self) -> int:
        """Return the dimensionality of vectors this model produces."""
        return self.model.get_sentence_embedding_dimension()
