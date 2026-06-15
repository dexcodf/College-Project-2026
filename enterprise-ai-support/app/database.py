"""Database engine, session factory, and declarative base.

Supports SQLite (zero-config dev) and PostgreSQL (production) transparently
via ``settings.database_url``. SQLite needs ``check_same_thread=False`` so the
connection can be shared across FastAPI's threadpool.
"""
from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

_is_sqlite = settings.database_url.startswith("sqlite")

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine, autoflush=False, autocommit=False, expire_on_commit=False
)


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models."""


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a scoped DB session and closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables. Models must be imported before calling this."""
    # Importing the models module registers them on Base.metadata.
    from app import models  # noqa: F401  (side-effect import)

    Base.metadata.create_all(bind=engine)
