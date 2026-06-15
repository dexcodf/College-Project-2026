"""Chat endpoints: ask questions, manage chats, submit feedback."""
from __future__ import annotations

import json

from fastapi import APIRouter, status
from pydantic import BaseModel

from app.dependencies import CurrentUser, DbSession
from app.exceptions import NotFoundError, PermissionError_
from app.models.feedback import Feedback, FeedbackRating
from app.models.message import Message
from app.schemas.chat import (
    AskRequest,
    AskResponse,
    ChatDetail,
    ChatOut,
    Citation,
    MessageOut,
)
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


def _to_message_out(msg: Message) -> MessageOut:
    citations = [Citation(**c) for c in json.loads(msg.citations or "[]")]
    return MessageOut(
        id=msg.id,
        role=msg.role,
        content=msg.content,
        citations=citations,
        latency_ms=msg.latency_ms,
        agent_route=msg.agent_route,
        created_at=msg.created_at,
    )


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, user: CurrentUser, db: DbSession) -> AskResponse:
    service = ChatService(db)
    assistant_msg = service.ask(
        user, payload.message, chat_id=payload.chat_id, top_k=payload.top_k
    )
    return AskResponse(
        chat_id=assistant_msg.chat_id, message=_to_message_out(assistant_msg)
    )


@router.get("/chats", response_model=list[ChatOut])
def list_chats(user: CurrentUser, db: DbSession) -> list[ChatOut]:
    service = ChatService(db)
    return [ChatOut.model_validate(c) for c in service.list_chats(user)]


@router.get("/chats/{chat_id}", response_model=ChatDetail)
def get_chat(chat_id: str, user: CurrentUser, db: DbSession) -> ChatDetail:
    service = ChatService(db)
    chat = service.get_chat(user, chat_id)
    return ChatDetail(
        id=chat.id,
        title=chat.title,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
        messages=[_to_message_out(m) for m in chat.messages],
    )


@router.delete("/chats/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chat(chat_id: str, user: CurrentUser, db: DbSession) -> None:
    ChatService(db).delete_chat(user, chat_id)


class FeedbackRequest(BaseModel):
    rating: FeedbackRating
    comment: str | None = None


@router.post("/messages/{message_id}/feedback", status_code=status.HTTP_201_CREATED)
def submit_feedback(
    message_id: str, payload: FeedbackRequest, user: CurrentUser, db: DbSession
) -> dict:
    message = db.get(Message, message_id)
    if message is None:
        raise NotFoundError("Message not found")
    if message.chat.user_id != user.id:
        raise PermissionError_("You cannot rate this message")
    feedback = Feedback(
        message_id=message_id,
        user_id=user.id,
        rating=payload.rating,
        comment=payload.comment,
    )
    db.add(feedback)
    db.commit()
    return {"detail": "Feedback recorded"}
