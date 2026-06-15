"""AnalyticsEvent model — append-only event log powering the dashboard."""
from __future__ import annotations

from sqlalchemy import Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.base import TimestampMixin, UUIDMixin


class AnalyticsEvent(UUIDMixin, TimestampMixin, Base):
    """A single tracked event, e.g. 'question_asked', 'document_uploaded',
    'login', 'feedback_given'. Numeric ``value`` carries metrics such as
    response latency; ``metadata_json`` carries arbitrary context."""

    __tablename__ = "analytics_events"

    event_type: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
