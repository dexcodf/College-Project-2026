"""Graph orchestration.

Builds a LangGraph ``StateGraph`` wiring supervisor → specialist → response.
If LangGraph is unavailable, an equivalent ``SequentialOrchestrator`` runs the
exact same nodes in the routed order, so behaviour is identical and the app
never hard-depends on the framework being importable.
"""
from __future__ import annotations

from app.agents.nodes import (
    analytics_node,
    faq_node,
    memory_node,
    response_node,
    retrieval_node,
    supervisor_node,
)
from app.agents.state import AgentState
from app.logging_config import get_logger

logger = get_logger("agents.graph")

_SPECIALISTS = {
    "retrieval": retrieval_node,
    "faq": faq_node,
    "memory": memory_node,
    "analytics": analytics_node,
}


def _route_selector(state: AgentState) -> str:
    return state.get("route", "retrieval")


class SequentialOrchestrator:
    """Framework-free fallback that mirrors the LangGraph control flow."""

    def invoke(self, state: AgentState) -> AgentState:
        state = {**state, **supervisor_node(state)}
        specialist = _SPECIALISTS.get(state["route"], retrieval_node)
        state = {**state, **specialist(state)}
        state = {**state, **response_node(state)}
        return state


def _build_langgraph():
    """Construct a compiled LangGraph state graph, or return None if absent."""
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception:  # pragma: no cover - optional dependency
        return None

    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    for name, node in _SPECIALISTS.items():
        graph.add_node(name, node)
    graph.add_node("response", response_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor", _route_selector, {name: name for name in _SPECIALISTS}
    )
    for name in _SPECIALISTS:
        graph.add_edge(name, "response")
    graph.add_edge("response", END)

    logger.info("langgraph_compiled")
    return graph.compile()


class AgentRunner:
    """Public entrypoint used by the chat service."""

    def __init__(self) -> None:
        self._graph = _build_langgraph()
        self._fallback = SequentialOrchestrator()
        self.backend = "langgraph" if self._graph is not None else "sequential"

    def run(self, state: AgentState) -> AgentState:
        if self._graph is not None:
            return self._graph.invoke(state)  # type: ignore[return-value]
        return self._fallback.invoke(state)


agent_runner = AgentRunner()
