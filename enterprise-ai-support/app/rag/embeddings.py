"""Embedding provider.

Primary: BAAI/bge-large-en-v1.5 via sentence-transformers. The bge models
recommend prefixing retrieval queries with an instruction; we apply that
automatically for queries (not for documents).

Fallback: if sentence-transformers / the model weights are unavailable
(e.g. offline CI), a deterministic hashing embedder keeps the pipeline
runnable so tests and demos don't hard-fail. The dimensionality is reported
via ``dimension`` so the vector store stays consistent.
"""
from __future__ import annotations

import hashlib
import math

from app.config import settings
from app.logging_config import get_logger

logger = get_logger("rag.embeddings")

_BGE_QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)
_FALLBACK_DIM = 384


class EmbeddingProvider:
    """Loads the embedding model lazily and exposes encode methods."""

    def __init__(self) -> None:
        self._model = None
        self._dim = _FALLBACK_DIM
        self._is_fallback = True
        self._load()

    def _load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                settings.embedding_model, device=settings.embedding_device
            )
            self._dim = self._model.get_sentence_embedding_dimension()
            self._is_fallback = False
            logger.info(
                "embedding_model_loaded",
                model=settings.embedding_model,
                dim=self._dim,
            )
        except Exception as exc:  # pragma: no cover - heavy/optional dependency
            logger.warning(
                "embedding_fallback",
                error=str(exc),
                note="using deterministic hashing embedder",
            )
            self._model = None
            self._is_fallback = True

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def is_fallback(self) -> bool:
        return self._is_fallback

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._model is not None:
            return self._model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            ).tolist()
        return [self._hash_embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        if self._model is not None:
            return self._model.encode(
                _BGE_QUERY_INSTRUCTION + text,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()
        return self._hash_embed(text)

    def _hash_embed(self, text: str) -> list[float]:
        """Deterministic bag-of-hashed-tokens embedding, L2-normalised."""
        vec = [0.0] * _FALLBACK_DIM
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % _FALLBACK_DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


# Singleton — model loading is expensive.
embedding_provider = EmbeddingProvider()
