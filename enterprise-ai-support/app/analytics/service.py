"""Analytics service: append events and compute dashboard aggregates."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.logging_config import get_logger
from app.models.analytics import AnalyticsEvent
from app.models.document import Document
from app.models.feedback import Feedback, FeedbackRating
from app.models.user import User
from app.schemas.analytics import AnalyticsOverview, TimeSeriesPoint

logger = get_logger("analytics")


def _as_utc(dt: datetime) -> datetime:
    """Normalise a (possibly naive) DB datetime to aware UTC.

    SQLite does not persist tzinfo, so timestamps come back naive; we treat
    them as UTC to keep comparisons valid across SQLite and PostgreSQL.
    """
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


class AnalyticsService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def track(
        self,
        event_type: str,
        *,
        user_id: str | None = None,
        value: float | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Append an analytics event (best-effort; never breaks the request)."""
        try:
            event = AnalyticsEvent(
                event_type=event_type,
                user_id=user_id,
                value=value,
                metadata_json=json.dumps(metadata or {}),
            )
            self.db.add(event)
            self.db.commit()
        except Exception as exc:  # pragma: no cover
            self.db.rollback()
            logger.warning("analytics_track_failed", error=str(exc))

    def overview(self, *, days: int = 14) -> AnalyticsOverview:
        since = datetime.now(timezone.utc) - timedelta(days=days)

        questions = self.db.scalars(
            select(AnalyticsEvent).where(AnalyticsEvent.event_type == "question_asked")
        ).all()

        total_questions = len(questions)
        total_users = self.db.scalar(select(func.count(User.id))) or 0
        total_documents = self.db.scalar(select(func.count(Document.id))) or 0

        # Active users in the last 7 days (distinct event authors).
        active_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        active_users_7d = (
            self.db.scalar(
                select(func.count(func.distinct(AnalyticsEvent.user_id))).where(
                    AnalyticsEvent.created_at >= active_cutoff,
                    AnalyticsEvent.user_id.is_not(None),
                )
            )
            or 0
        )

        latencies = [q.value for q in questions if q.value is not None]
        avg_response_ms = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

        # Feedback rate.
        up = self.db.scalar(
            select(func.count(Feedback.id)).where(Feedback.rating == FeedbackRating.UP)
        ) or 0
        total_fb = self.db.scalar(select(func.count(Feedback.id))) or 0
        positive_rate = round(up / total_fb, 3) if total_fb else 0.0

        # Time series (per day) for questions + avg response time.
        per_day_count: dict[str, int] = defaultdict(int)
        per_day_latency: dict[str, list[float]] = defaultdict(list)
        routes: Counter[str] = Counter()
        for q in questions:
            created = _as_utc(q.created_at)
            if created < since:
                continue
            day = created.date().isoformat()
            per_day_count[day] += 1
            if q.value is not None:
                per_day_latency[day].append(q.value)
            try:
                route = json.loads(q.metadata_json).get("route")
                if route:
                    routes[route] += 1
            except json.JSONDecodeError:
                pass

        questions_over_time = [
            TimeSeriesPoint(date=d, value=c) for d, c in sorted(per_day_count.items())
        ]
        response_times_over_time = [
            TimeSeriesPoint(date=d, value=round(sum(v) / len(v), 2))
            for d, v in sorted(per_day_latency.items())
        ]

        return AnalyticsOverview(
            total_questions=total_questions,
            total_users=total_users,
            active_users_7d=active_users_7d,
            total_documents=total_documents,
            avg_response_ms=avg_response_ms,
            positive_feedback_rate=positive_rate,
            questions_over_time=questions_over_time,
            response_times_over_time=response_times_over_time,
            top_routes=dict(routes),
        )
