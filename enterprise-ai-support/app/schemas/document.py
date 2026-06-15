"""Document schemas."""
from __future__ import annotations

from datetime import datetime

from app.models.document import DocumentStatus
from app.schemas.common import ORMModel


class DocumentOut(ORMModel):
    id: str
    filename: str
    content_type: str
    size_bytes: int
    status: DocumentStatus
    chunk_count: int
    error: str | None = None
    created_at: datetime


class IngestResult(ORMModel):
    document: DocumentOut
    chunks_created: int
