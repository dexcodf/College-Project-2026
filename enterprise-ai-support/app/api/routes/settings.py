"""Settings & profile endpoints."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.config import settings as app_settings
from app.dependencies import CurrentUser, DbSession
from app.schemas.auth import UserOut

router = APIRouter(prefix="/settings", tags=["settings"])


class PublicSettings(BaseModel):
    app_name: str
    environment: str
    embedding_model: str
    llm_model: str
    retrieval_top_k: int


@router.get("/public", response_model=PublicSettings)
def public_settings() -> PublicSettings:
    """Non-sensitive runtime configuration for the frontend."""
    return PublicSettings(
        app_name=app_settings.app_name,
        environment=app_settings.environment,
        embedding_model=app_settings.embedding_model,
        llm_model=app_settings.llm_model,
        retrieval_top_k=app_settings.retrieval_top_k,
    )


class ProfileUpdate(BaseModel):
    full_name: str = Field(min_length=1, max_length=255)


@router.put("/profile", response_model=UserOut)
def update_profile(payload: ProfileUpdate, user: CurrentUser, db: DbSession) -> UserOut:
    user.full_name = payload.full_name
    db.commit()
    db.refresh(user)
    return UserOut.model_validate(user)
