"""FastAPI application factory and entrypoint.

Run with: ``uvicorn app.main:app --reload``
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import api_router
from app.auth.service import AuthService
from app.config import settings
from app.database import SessionLocal, init_db
from app.exceptions import install_exception_handlers
from app.logging_config import configure_logging, get_logger
from app.middleware import RequestContextMiddleware

configure_logging()
logger = get_logger("main")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Startup/shutdown: create schema and bootstrap a default admin."""
    init_db()
    with SessionLocal() as db:
        AuthService(db).ensure_default_admin(
            email="admin@example.com", password="admin12345"
        )
    logger.info(
        "startup_complete",
        app=settings.app_name,
        environment=settings.environment,
    )
    yield
    logger.info("shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="Enterprise AI Customer Support Assistant — RAG + agents + voice.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [settings.backend_url],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestContextMiddleware)

    install_exception_handlers(app)
    app.include_router(api_router)

    @app.get("/", tags=["health"])
    def root() -> dict:
        return {"name": settings.app_name, "version": __version__, "docs": "/docs"}

    return app


app = create_app()
