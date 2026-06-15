"""Integration tests across the chat + documents API."""
from __future__ import annotations


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_full_chat_flow(client, auth_headers):
    # Upload a small text document.
    res = client.post(
        "/api/documents",
        headers=auth_headers,
        files={"file": ("faq.txt", b"Our return window is 30 days for all products.", "text/plain")},
    )
    assert res.status_code == 201, res.text
    assert res.json()["document"]["status"] == "ready"

    # Ask a question.
    res = client.post(
        "/api/chat/ask",
        headers=auth_headers,
        json={"message": "What is the return window?"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["message"]["content"]
    assert "chat_id" in body

    # The chat shows up in the listing.
    res = client.get("/api/chat/chats", headers=auth_headers)
    assert res.status_code == 200
    assert len(res.json()) >= 1


def test_feedback_flow(client, auth_headers):
    ask = client.post(
        "/api/chat/ask", headers=auth_headers, json={"message": "hello"}
    ).json()
    message_id = ask["message"]["id"]
    res = client.post(
        f"/api/chat/messages/{message_id}/feedback",
        headers=auth_headers,
        json={"rating": "up"},
    )
    assert res.status_code == 201


def test_analytics_requires_admin(client, auth_headers):
    # A normal user is forbidden.
    assert client.get("/api/analytics/overview", headers=auth_headers).status_code == 403


def test_documents_listed_per_user(client, auth_headers):
    res = client.get("/api/documents", headers=auth_headers)
    assert res.status_code == 200
    assert isinstance(res.json(), list)
