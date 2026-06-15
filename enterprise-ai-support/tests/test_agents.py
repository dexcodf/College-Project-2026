"""Agent routing and orchestration."""
from __future__ import annotations

from app.agents.graph import SequentialOrchestrator, agent_runner
from app.agents.nodes import supervisor_node
from app.agents.tools import lookup_faq


def test_supervisor_routes_faq():
    state = {"query": "how do I reset password"}
    assert supervisor_node(state)["route"] == "faq"


def test_supervisor_routes_analytics():
    state = {"query": "how many documents are indexed"}
    assert supervisor_node(state)["route"] == "analytics"


def test_supervisor_defaults_to_retrieval():
    state = {"query": "what is the warranty on product X"}
    assert supervisor_node(state)["route"] == "retrieval"


def test_faq_lookup():
    assert lookup_faq("how do I reset password") is not None
    assert lookup_faq("totally unrelated gibberish") is None


def test_sequential_orchestrator_produces_answer():
    orch = SequentialOrchestrator()
    result = orch.invoke(
        {"query": "how do I reset password", "owner_id": "u1", "top_k": 4, "history": []}
    )
    assert result["answer"]
    assert result["route"] == "faq"


def test_agent_runner_backend_is_known():
    assert agent_runner.backend in {"langgraph", "sequential"}
