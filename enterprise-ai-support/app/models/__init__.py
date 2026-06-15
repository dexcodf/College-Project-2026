"""SQLAlchemy ORM models.

Importing this package registers every model on ``Base.metadata`` so that
``init_db()`` can create the full schema in one call.
"""
from app.models.analytics import AnalyticsEvent
from app.models.chat import Chat
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.models.feedback import Feedback, FeedbackRating
from app.models.message import Message, MessageRole
from app.models.session import UserSession
from app.models.user import Role, User

__all__ = [
    "AnalyticsEvent",
    "Chat",
    "Document",
    "DocumentChunk",
    "DocumentStatus",
    "Feedback",
    "FeedbackRating",
    "Message",
    "MessageRole",
    "UserSession",
    "Role",
    "User",
]
