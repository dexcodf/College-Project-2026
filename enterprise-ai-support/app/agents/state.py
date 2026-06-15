"""Shared graph state passed between agent nodes."""
from __future__ import annotations

from typing import TypedDict

from app.rag.vector_store import RetrievedChunk
from app.schemas.chat import Citation


class AgentState(TypedDict, total=False):
    """Mutable state threaded through the agent graph.

    Nodes read the fields they need and write their outputs back, so the graph
    is a pure data-flow with no hidden side channels.
    """

    # ---- inputs ----
    query: str
    owner_id: str
    chat_id: str | None
    history: list[dict[str, str]]
    top_k: int

    # ---- routing ----
    route: str          # retrieval | faq | memory | analytics
    route_reason: str

    # ---- retrieval ----
    chunks: list[RetrievedChunk]
    context: str

    # ---- output ----
    answer: str
    citations: list[Citation]
