"""API routers, aggregated into a single APIRouter."""
from fastapi import APIRouter

from app.api.routes import (
    analytics,
    auth,
    chat,
    documents,
    health,
    memory,
    settings as settings_routes,
    voice,
)

api_router = APIRouter(prefix="/api")
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(documents.router)
api_router.include_router(chat.router)
api_router.include_router(analytics.router)
api_router.include_router(memory.router)
api_router.include_router(settings_routes.router)
api_router.include_router(voice.router)
