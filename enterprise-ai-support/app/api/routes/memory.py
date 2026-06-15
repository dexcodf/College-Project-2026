"""Memory endpoints: read/update persistent user preferences."""
from __future__ import annotations

from fastapi import APIRouter

from app.dependencies import CurrentUser, DbSession
from app.memory.store import UserMemory
from app.schemas.auth import UserPreferences

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("/preferences", response_model=UserPreferences)
def get_preferences(user: CurrentUser, db: DbSession) -> UserPreferences:
    prefs = UserMemory(db).get_preferences(user)
    return UserPreferences(**{**UserPreferences().model_dump(), **prefs})


@router.put("/preferences", response_model=UserPreferences)
def update_preferences(
    payload: UserPreferences, user: CurrentUser, db: DbSession
) -> UserPreferences:
    prefs = UserMemory(db).update_preferences(user, payload.model_dump())
    return UserPreferences(**prefs)
