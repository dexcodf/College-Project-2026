"""Tools available to agents.

Tools are plain callables with typed signatures so they can be exposed to a
tool-calling LLM or invoked directly by graph nodes. Keeping them framework-
agnostic makes them trivial to unit-test.
"""
from __future__ import annotations

from app.rag.retriever import retrieve
from app.rag.vector_store import RetrievedChunk, vector_store

# A small, curated FAQ the FAQ agent can answer instantly without retrieval.
FAQ: dict[str, str] = {
    "hours": "Our support team is available 24/7 through this assistant. Human "
    "agents are online Monday–Friday, 9am–6pm.",
    "reset password": "To reset your password, open Settings → Security and "
    "choose “Reset password”. You'll receive a confirmation email.",
    "contact": "You can reach a human agent any time by typing “talk to an "
    "agent”, or email support@example.com.",
    "refund": "Refunds are processed within 5–7 business days to your original "
    "payment method. See our refund policy for eligibility.",
}


def search_knowledge_base(
    query: str, *, owner_id: str | None, top_k: int
) -> list[RetrievedChunk]:
    """Semantic + hybrid search over the document knowledge base."""
    return retrieve(query, owner_id=owner_id, top_k=top_k)


def lookup_faq(query: str) -> str | None:
    """Return a canned FAQ answer if the query clearly matches one."""
    q = query.lower()
    for key, answer in FAQ.items():
        if all(word in q for word in key.split()):
            return answer
    return None


def knowledge_base_size() -> int:
    """Number of indexed chunks — used by the analytics/database agent."""
    return vector_store.count()
