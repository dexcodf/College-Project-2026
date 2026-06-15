"""Lightweight cross-encoder-style re-ranker.

Re-ranks candidate chunks against the query by blending the dense retrieval
score with lexical overlap signals (token overlap + phrase containment). This
is a dependency-free approximation of a cross-encoder that meaningfully
improves ordering without a second heavyweight model; swap in
``cross-encoder/ms-marco-MiniLM`` here for production-grade re-ranking.
"""
from __future__ import annotations

import re

from app.rag.vector_store import RetrievedChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def rerank(
    query: str, candidates: list[RetrievedChunk], *, top_n: int
) -> list[RetrievedChunk]:
    """Return the ``top_n`` candidates ordered by a blended relevance score."""
    if not candidates:
        return []

    q_tokens = _tokens(query)
    q_lower = query.lower()

    rescored: list[tuple[float, RetrievedChunk]] = []
    for chunk in candidates:
        c_tokens = _tokens(chunk.text)
        overlap = len(q_tokens & c_tokens) / (len(q_tokens) or 1)
        phrase_bonus = 0.15 if q_lower in chunk.text.lower() else 0.0
        # Blend: dense similarity dominates, lexical signals refine ties.
        blended = 0.7 * chunk.score + 0.3 * overlap + phrase_bonus
        chunk.score = round(blended, 4)
        rescored.append((blended, chunk))

    rescored.sort(key=lambda pair: pair[0], reverse=True)
    return [chunk for _, chunk in rescored[:top_n]]
