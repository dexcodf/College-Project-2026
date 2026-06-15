"""Chat / message / citation schemas."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.models.message import MessageRole
from app.schemas.common import ORMModel


class Citation(BaseModel):
    """A single source reference attached to an assistant answer."""

    document_id: str
    filename: str
    page: int | None = None
    score: float
    snippet: str


class MessageOut(ORMModel):
    id: str
    role: MessageRole
    content: str
    citations: list[Citation] = Field(default_factory=list)
    latency_ms: float | None = None
    agent_route: str | None = None
    created_at: datetime


class ChatOut(ORMModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ChatDetail(ChatOut):
    messages: list[MessageOut] = Field(default_factory=list)


class AskRequest(BaseModel):
    """Ask a question, optionally within an existing chat."""

    message: str = Field(min_length=1, max_length=8000)
    chat_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class AskResponse(BaseModel):
    chat_id: str
    message: MessageOut
