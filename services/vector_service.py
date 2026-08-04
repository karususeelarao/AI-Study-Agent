"""
services/vector_service.py
==============================================================================
FAISS-backed vector store service for Retrieval-Augmented Generation (RAG).

Responsibilities:
    - Build/maintain a FAISS index of embedded PDF chunks.
    - Persist the index + associated text/metadata to disk so it survives
      app restarts.
    - Provide similarity search for RAG-based question answering.

Design notes:
    - Uses IndexFlatIP (inner product) over L2-normalized vectors, which
      is mathematically equivalent to cosine similarity search but faster.
    - Since raw FAISS indexes only store vectors (not text), we keep a
      parallel Python list `self.texts` / `self.metadata` and persist it
      alongside the index as a pickle file. This "index + sidecar store"
      pattern is the standard way to use FAISS for RAG.
==============================================================================
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from config import settings
from logger import get_logger
from services.embedding_service import EmbeddingService

logger = get_logger(__name__)


class VectorServiceError(Exception):
    """Raised when a vector store operation fails."""


@dataclass
class RetrievedChunk:
    """A single retrieved chunk of text with its similarity score."""

    text: str
    score: float
    metadata: dict


class VectorService:
    """
    Manages a FAISS index for one document collection (e.g., one uploaded
    PDF, or one session's set of uploaded PDFs).
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        index_dir: Path | None = None,
        index_name: str | None = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.index_dir = Path(index_dir or settings.vector_db_dir)
        self.index_name = index_name or settings.faiss_index_name
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.dimension = self.embedding_service.get_embedding_dimension()
        self.index: faiss.Index = faiss.IndexFlatIP(self.dimension)
        self.texts: list[str] = []
        self.metadata: list[dict] = []

        logger.info(
            "VectorService initialized (dim=%d, index_dir=%s, index_name=%s)",
            self.dimension,
            self.index_dir,
            self.index_name,
        )

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------
    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> int:
        """
        Embed and add a batch of text chunks to the index.

        Args:
            texts: Raw text chunks (e.g. from a PDF splitter).
            metadatas: Optional per-chunk metadata (e.g. {"source": "file.pdf", "page": 3}).

        Returns:
            Number of vectors added.
        """
        if not texts:
            raise VectorServiceError("No texts provided to index.")
        metadatas = metadatas or [{} for _ in texts]
        if len(metadatas) != len(texts):
            raise VectorServiceError("texts and metadatas must be the same length.")

        try:
            vectors = self.embedding_service.embed_texts(texts)
            self.index.add(vectors)
            self.texts.extend(texts)
            self.metadata.extend(metadatas)
            logger.info("Added %d vectors to FAISS index (total=%d)", len(texts), self.index.ntotal)
            return len(texts)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to add texts to FAISS index")
            raise VectorServiceError(f"Failed to add texts to index: {exc}") from exc

    def similarity_search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """
        Retrieve the top_k most similar chunks to a query.

        Returns an empty list (not an error) if the index has no vectors,
        so callers can gracefully fall back to "no context available".
        """
        top_k = top_k or settings.retrieval_top_k
        if self.index.ntotal == 0:
            logger.warning("similarity_search called on empty index.")
            return []
        try:
            query_vector = self.embedding_service.embed_query(query).reshape(1, -1)
            k = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(query_vector, k)
            results: list[RetrievedChunk] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                results.append(
                    RetrievedChunk(
                        text=self.texts[idx],
                        score=float(score),
                        metadata=self.metadata[idx],
                    )
                )
            return results
        except Exception as exc:  # noqa: BLE001
            logger.exception("Similarity search failed for query=%r", query)
            raise VectorServiceError(f"Similarity search failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self) -> None:
        """Persist the FAISS index and its sidecar text/metadata store to disk."""
        try:
            index_path = self.index_dir / f"{self.index_name}.faiss"
            sidecar_path = self.index_dir / f"{self.index_name}.pkl"
            faiss.write_index(self.index, str(index_path))
            with open(sidecar_path, "wb") as f:
                pickle.dump({"texts": self.texts, "metadata": self.metadata}, f)
            logger.info("Saved FAISS index (%d vectors) to %s", self.index.ntotal, index_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to save FAISS index")
            raise VectorServiceError(f"Failed to save index: {exc}") from exc

    def load(self) -> bool:
        """
        Load a previously saved index from disk, if present.

        Returns:
            True if an index was found and loaded, False otherwise.
        """
        index_path = self.index_dir / f"{self.index_name}.faiss"
        sidecar_path = self.index_dir / f"{self.index_name}.pkl"
        if not index_path.exists() or not sidecar_path.exists():
            logger.info("No existing FAISS index found at %s; starting fresh.", index_path)
            return False
        try:
            self.index = faiss.read_index(str(index_path))
            with open(sidecar_path, "rb") as f:
                sidecar = pickle.load(f)
            self.texts = sidecar["texts"]
            self.metadata = sidecar["metadata"]
            logger.info("Loaded FAISS index with %d vectors from %s", self.index.ntotal, index_path)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load FAISS index")
            raise VectorServiceError(f"Failed to load index: {exc}") from exc

    def reset(self) -> None:
        """Clear the in-memory index (used when a new PDF replaces the old context)."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.texts = []
        self.metadata = []
        logger.info("VectorService index reset.")

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        return self.index.ntotal
