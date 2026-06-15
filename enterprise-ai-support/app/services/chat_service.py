"""Chat service — orchestrates a full Q&A turn.

Ties together conversation memory, the agent graph, citation persistence, and
analytics tracking. This is the single entry point the API route calls.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.graph import agent_runner
from app.agents.state import AgentState
from app.analytics.service import AnalyticsService
from app.config import settings
from app.exceptions import NotFoundError, PermissionError_
from app.logging_config import get_logger
from app.memory.store import ConversationMemory
from app.models.chat import Chat
from app.models.message import Message, MessageRole
from app.models.user import User
from app.schemas.chat import Citation

logger = get_logger("chat.service")


class ChatService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.memory = ConversationMemory(db)
        self.analytics = AnalyticsService(db)

    # ---- chat lifecycle ----
    def get_or_create_chat(self, user: User, chat_id: str | None, first_msg: str) -> Chat:
        if chat_id:
            chat = self.db.get(Chat, chat_id)
            if chat is None:
                raise NotFoundError("Chat not found")
            if chat.user_id != user.id:
                raise PermissionError_("You do not have access to this chat")
            return chat
        title = (first_msg[:48] + "…") if len(first_msg) > 48 else first_msg
        chat = Chat(user_id=user.id, title=title or "New conversation")
        self.db.add(chat)
        self.db.commit()
        self.db.refresh(chat)
        return chat

    def list_chats(self, user: User) -> list[Chat]:
        return list(
            self.db.scalars(
                select(Chat)
                .where(Chat.user_id == user.id)
                .order_by(Chat.updated_at.desc())
            ).all()
        )

    def get_chat(self, user: User, chat_id: str) -> Chat:
        chat = self.db.get(Chat, chat_id)
        if chat is None:
            raise NotFoundError("Chat not found")
        if chat.user_id != user.id:
            raise PermissionError_("You do not have access to this chat")
        return chat

    def delete_chat(self, user: User, chat_id: str) -> None:
        chat = self.get_chat(user, chat_id)
        self.db.delete(chat)
        self.db.commit()

    # ---- the main turn ----
    def ask(
        self, user: User, message_text: str, *, chat_id: str | None, top_k: int | None
    ) -> Message:
        chat = self.get_or_create_chat(user, chat_id, message_text)

        user_msg = Message(
            chat_id=chat.id, role=MessageRole.USER, content=message_text
        )
        self.db.add(user_msg)
        self.db.commit()

        history = self.memory.recent_turns(chat.id)

        state: AgentState = {
            "query": message_text,
            "owner_id": user.id,
            "chat_id": chat.id,
            "history": history,
            "top_k": top_k or settings.retrieval_top_k,
        }

        start = time.perf_counter()
        result = agent_runner.run(state)
        latency_ms = (time.perf_counter() - start) * 1000

        citations: list[Citation] = result.get("citations", []) or []
        assistant_msg = Message(
            chat_id=chat.id,
            role=MessageRole.ASSISTANT,
            content=result.get("answer", "") or "I'm sorry, I couldn't generate a response.",
            citations=json.dumps([c.model_dump() for c in citations]),
            latency_ms=round(latency_ms, 2),
            agent_route=result.get("route"),
        )
        self.db.add(assistant_msg)
        # Touch chat.updated_at so it sorts to the top of the conversation list.
        chat.updated_at = datetime.now(timezone.utc)
        self.db.commit()
        self.db.refresh(assistant_msg)

        self.analytics.track(
            "question_asked",
            user_id=user.id,
            value=latency_ms,
            metadata={"route": result.get("route"), "backend": agent_runner.backend},
        )
        logger.info(
            "ask_done",
            chat_id=chat.id,
            route=result.get("route"),
            latency_ms=round(latency_ms, 2),
        )
        return assistant_msg
