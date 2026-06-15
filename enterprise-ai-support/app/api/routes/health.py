"""Health and readiness endpoints (for load balancers / k8s probes)."""
from __future__ import annotations

from fastapi import APIRouter

from app.agents.graph import agent_runner
from app.config import settings
from app.rag.embeddings import embedding_provider
from app.services.llm import llm_client

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    """Liveness + capability report."""
    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.environment,
        "capabilities": {
            "llm": llm_client.available,
            "embeddings": "model" if not embedding_provider.is_fallback else "fallback",
            "agent_backend": agent_runner.backend,
        },
    }
