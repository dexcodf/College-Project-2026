"""Shared pytest fixtures.

Uses an isolated temporary SQLite database per test session and overrides the
``get_db`` dependency so tests never touch the dev database.
"""
from __future__ import annotations

import os
import tempfile
from collections.abc import Generator

import pytest

# Force a throwaway SQLite DB *before* importing app modules that read settings.
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp_db.name}"
os.environ["SECRET_KEY"] = "test-secret-key-for-pytest"
os.environ["LLM_API_KEY"] = ""  # exercise the offline fallback path

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.database import Base, get_db  # noqa: E402
from app.main import app  # noqa: E402

engine = create_engine(
    os.environ["DATABASE_URL"], connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@pytest.fixture(scope="session", autouse=True)
def _create_schema() -> Generator[None, None, None]:
    import app.models  # noqa: F401  (register models)

    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def _override_get_db() -> Generator:
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = _override_get_db


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def auth_headers(client: TestClient) -> dict[str, str]:
    """Register a fresh user and return Authorization headers."""
    import uuid

    email = f"user_{uuid.uuid4().hex[:8]}@example.com"
    res = client.post(
        "/api/auth/signup",
        json={"email": email, "password": "password123", "full_name": "Test User"},
    )
    assert res.status_code == 201, res.text
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
