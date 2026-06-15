"""Analytics schemas for the dashboard."""
from __future__ import annotations

from pydantic import BaseModel


class TimeSeriesPoint(BaseModel):
    date: str  # ISO date
    value: float


class AnalyticsOverview(BaseModel):
    total_questions: int
    total_users: int
    active_users_7d: int
    total_documents: int
    avg_response_ms: float
    positive_feedback_rate: float  # 0..1

    questions_over_time: list[TimeSeriesPoint]
    response_times_over_time: list[TimeSeriesPoint]
    top_routes: dict[str, int]
