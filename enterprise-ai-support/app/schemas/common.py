"""Shared schema primitives."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ORMModel(BaseModel):
    """Base for schemas read from ORM objects."""

    model_config = ConfigDict(from_attributes=True)


class Message(BaseModel):
    """Generic message envelope for simple acknowledgements."""

    detail: str


class TimestampedSchema(ORMModel):
    id: str
    created_at: datetime
    updated_at: datetime
