"""Agent node implementations.

Each function is a node in the graph: it takes ``AgentState`` and returns a
partial state update. They are deliberately framework-neutral so the same
nodes power both the LangGraph build and the sequential fallback.
"""
from __future__ import annotations

from app.agents.state import AgentState
from app.agents.tools import lookup_faq, search_knowledge_base
from app.logging_config import get_logger
from app.rag.citations import build_citations, format_context
from app.services.llm import llm_client

logger = get_logger("agents")

SYSTEM_PROMPT = (
    "You are an enterprise customer-support assistant. Answer using ONLY the "
    "provided context when it is relevant. Cite sources inline as [1], [2] "
    "matching the numbered context. If the context is insufficient, say so "
    "honestly and suggest next steps. Be concise, accurate, and professional."
)


# ---------------------------------------------------------------- supervisor
def supervisor_node(state: AgentState) -> AgentState:
    """Route the query to a specialist agent using cheap heuristics.

    Heuristics keep routing deterministic and fast; an LLM router can be
    swapped in without changing the graph shape.
    """
    query = state["query"].lower()

    if lookup_faq(query) is not None:
        route, reason = "faq", "matched a known FAQ pattern"
    elif any(w in query for w in ("remember", "my name is", "i prefer", "last time")):
        route, reason = "memory", "references personal/conversational memory"
    elif any(w in query for w in ("how many", "stats", "analytics", "usage")):
        route, reason = "analytics", "asks about platform metrics"
    else:
        route, reason = "retrieval", "general knowledge-base question"

    logger.info("route", route=route, reason=reason)
    return {"route": route, "route_reason": reason}


# ---------------------------------------------------------------- retrieval
def retrieval_node(state: AgentState) -> AgentState:
    chunks = search_knowledge_base(
        state["query"], owner_id=state.get("owner_id"), top_k=state.get("top_k", 8)
    )
    return {
        "chunks": chunks,
        "context": format_context(chunks),
        "citations": build_citations(chunks),
    }


# ---------------------------------------------------------------- faq
def faq_node(state: AgentState) -> AgentState:
    answer = lookup_faq(state["query"]) or ""
    return {"answer": answer, "chunks": [], "citations": []}


# ---------------------------------------------------------------- memory
def memory_node(state: AgentState) -> AgentState:
    """Memory questions still benefit from history; retrieval is skipped and
    the response node leans on conversation history injected upstream."""
    return {"chunks": [], "context": "", "citations": []}


# ---------------------------------------------------------------- analytics
def analytics_node(state: AgentState) -> AgentState:
    from app.agents.tools import knowledge_base_size

    size = knowledge_base_size()
    answer = (
        f"The knowledge base currently contains {size} indexed passages. "
        "Open the Analytics dashboard for usage, response-time, and feedback "
        "metrics."
    )
    return {"answer": answer, "chunks": [], "citations": []}


# ---------------------------------------------------------------- response
def response_node(state: AgentState) -> AgentState:
    """Synthesise the final answer with the LLM, grounded in context+history."""
    # FAQ / analytics nodes may have already produced a direct answer.
    if state.get("answer"):
        return {"answer": state["answer"]}

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = state.get("context", "")
    if context:
        messages.append({"role": "system", "content": f"CONTEXT:\n{context}"})
    messages.extend(state.get("history", []))
    messages.append({"role": "user", "content": state["query"]})

    answer = llm_client.complete(messages)
    return {"answer": answer}
