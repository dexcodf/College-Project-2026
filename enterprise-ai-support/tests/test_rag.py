"""RAG: embeddings, vector store, retrieval, reranking, citations."""
from __future__ import annotations

from app.rag.citations import build_citations, format_context
from app.rag.embeddings import embedding_provider
from app.rag.reranker import rerank
from app.rag.vector_store import RetrievedChunk, VectorStore


def test_embeddings_have_consistent_dimension():
    vecs = embedding_provider.embed_documents(["hello world", "another doc"])
    assert len(vecs) == 2
    assert len(vecs[0]) == embedding_provider.dimension
    assert len(vecs[0]) == len(vecs[1])


def test_vector_store_add_and_query_roundtrip():
    store = VectorStore()
    store._collection = None  # force the in-memory path for a deterministic test
    store.add(
        ids=["v1", "v2"],
        texts=["password reset instructions", "billing and refund policy"],
        metadatas=[
            {"document_id": "d1", "owner_id": "u1", "filename": "a.txt", "page": 1},
            {"document_id": "d2", "owner_id": "u1", "filename": "b.txt", "page": 1},
        ],
    )
    results = store.query("how do I reset my password", top_k=2, owner_id="u1")
    assert results
    assert results[0].metadata["document_id"] in {"d1", "d2"}


def test_reranker_orders_by_relevance():
    candidates = [
        RetrievedChunk("1", "the cat sat on the mat", 0.4, {}),
        RetrievedChunk("2", "reset your password from settings", 0.5, {}),
    ]
    ranked = rerank("how to reset password", candidates, top_n=2)
    assert ranked[0].vector_id == "2"


def test_build_citations_dedupes():
    chunks = [
        RetrievedChunk("1", "text one", 0.9, {"document_id": "d1", "filename": "a", "page": 1}),
        RetrievedChunk("2", "text two", 0.8, {"document_id": "d1", "filename": "a", "page": 1}),
    ]
    citations = build_citations(chunks)
    assert len(citations) == 1


def test_format_context_numbers_sources():
    chunks = [
        RetrievedChunk("1", "alpha", 0.9, {"filename": "a.txt", "page": 2}),
    ]
    ctx = format_context(chunks)
    assert "[1]" in ctx and "a.txt" in ctx
