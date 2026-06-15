"""Analytics endpoint (admin-only dashboard data)."""
from __future__ import annotations

from fastapi import APIRouter

from app.analytics.service import AnalyticsService
from app.dependencies import AdminUser, DbSession
from app.schemas.analytics import AnalyticsOverview

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/overview", response_model=AnalyticsOverview)
def overview(_: AdminUser, db: DbSession, days: int = 14) -> AnalyticsOverview:
    return AnalyticsService(db).overview(days=days)
