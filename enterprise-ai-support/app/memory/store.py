"""Memory store: short-term conversation history + persistent user prefs.

- **Conversation memory**: the recent turns of a chat, loaded from the DB and
  trimmed to a token-ish budget for prompting.
- **Persistent memory**: per-user preferences stored as JSON on the User row.
"""
from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.logging_config import get_logger
from app.models.message import Message, MessageRole
from app.models.user import User

logger = get_logger("memory")

# Approximate budget for recalled history (characters as a token proxy).
_HISTORY_CHAR_BUDGET = 4000


class ConversationMemory:
    """Loads and formats recent chat history for prompting."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def recent_turns(self, chat_id: str, *, limit: int = 12) -> list[dict[str, str]]:
        """Return the most recent turns as OpenAI-style messages (chronological)."""
        rows = self.db.scalars(
            select(Message)
            .where(Message.chat_id == chat_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        ).all()
        rows = list(reversed(rows))

        messages: list[dict[str, str]] = []
        budget = _HISTORY_CHAR_BUDGET
        for msg in rows:
            content = msg.content
            if budget - len(content) < 0:
                break
            budget -= len(content)
            role = "assistant" if msg.role == MessageRole.ASSISTANT else "user"
            messages.append({"role": role, "content": content})
        return messages


class UserMemory:
    """Persistent per-user preferences and recalled facts."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_preferences(self, user: User) -> dict:
        try:
            return json.loads(user.preferences or "{}")
        except json.JSONDecodeError:
            return {}

    def update_preferences(self, user: User, updates: dict) -> dict:
        prefs = self.get_preferences(user)
        prefs.update(updates)
        user.preferences = json.dumps(prefs)
        self.db.commit()
        logger.info("preferences_updated", user_id=user.id, keys=list(updates))
        return prefs
