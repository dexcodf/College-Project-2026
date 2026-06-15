"""Retrieval engine: dense retrieval + hybrid lexical fusion + re-ranking.

Pipeline per query:
  1. (optional) query rewriting / expansion
  2. dense retrieval from the vector store (top_k)
  3. BM25 lexical fusion over the dense candidate pool (Reciprocal Rank Fusion)
  4. cross-encoder-style re-ranking down to top_n
  5. context compression (dedupe + length cap)
"""
from __future__ import annotations

from app.config import settings
from app.logging_config import get_logger
from app.rag.reranker import rerank
from app.rag.vector_store import RetrievedChunk, vector_store

logger = get_logger("rag.retriever")


def rewrite_query(query: str) -> str:
    """Cheap, deterministic query normalisation/expansion.

    Strips filler, collapses whitespace. A production system can replace this
    with an LLM-based rewriter; the interface stays the same.
    """
    cleaned = " ".join(query.split())
    return cleaned


def _bm25_fuse(query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Fuse dense ranking with BM25 lexical ranking via Reciprocal Rank Fusion."""
    if len(candidates) < 2:
        return candidates
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:  # pragma: no cover
        return candidates

    corpus = [c.text.lower().split() for c in candidates]
    bm25 = BM25Okapi(corpus)
    lexical_scores = bm25.get_scores(query.lower().split())

    dense_rank = {c.vector_id: r for r, c in enumerate(candidates)}
    lexical_order = sorted(
        range(len(candidates)), key=lambda i: lexical_scores[i], reverse=True
    )
    lexical_rank = {candidates[i].vector_id: r for r, i in enumerate(lexical_order)}

    k = 60  # standard RRF constant
    fused = []
    for c in candidates:
        rrf = 1.0 / (k + dense_rank[c.vector_id]) + 1.0 / (k + lexical_rank[c.vector_id])
        fused.append((rrf, c))
    fused.sort(key=lambda pair: pair[0], reverse=True)
    return [c for _, c in fused]


def _compress(chunks: list[RetrievedChunk], *, max_chars: int = 6000) -> list[RetrievedChunk]:
    """Drop near-duplicate chunks and cap total context length."""
    seen: set[str] = set()
    out: list[RetrievedChunk] = []
    total = 0
    for chunk in chunks:
        key = chunk.text[:120]
        if key in seen:
            continue
        seen.add(key)
        if total + len(chunk.text) > max_chars and out:
            break
        out.append(chunk)
        total += len(chunk.text)
    return out


def retrieve(
    query: str,
    *,
    owner_id: str | None = None,
    top_k: int | None = None,
    top_n: int | None = None,
) -> list[RetrievedChunk]:
    """Run the full retrieval pipeline and return ranked, compressed chunks."""
    top_k = top_k or settings.retrieval_top_k
    top_n = top_n or settings.rerank_top_n

    rewritten = rewrite_query(query)
    dense = vector_store.query(rewritten, top_k=top_k, owner_id=owner_id)
    if not dense:
        logger.info("retrieve_empty", query=query[:80])
        return []

    fused = _bm25_fuse(rewritten, dense)
    reranked = rerank(rewritten, fused, top_n=top_n)
    compressed = _compress(reranked)
    logger.info(
        "retrieve_done",
        candidates=len(dense),
        returned=len(compressed),
    )
    return compressed
