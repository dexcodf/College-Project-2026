"""Typed HTTP client for the backend API.

A thin wrapper over ``requests`` that injects the bearer token, raises a
friendly ``APIError`` on failures, and exposes one method per backend
operation the UI needs.
"""
from __future__ import annotations

import os
from typing import Any

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
TIMEOUT = 120


class APIError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class APIClient:
    def __init__(self, base_url: str = BACKEND_URL, token: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token

    # ---- internals ----
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        headers = {**self._headers(), **kwargs.pop("headers", {})}
        try:
            resp = requests.request(
                method, url, headers=headers, timeout=TIMEOUT, **kwargs
            )
        except requests.RequestException as exc:
            raise APIError(f"Cannot reach backend at {self.base_url}: {exc}") from exc

        if resp.status_code >= 400:
            message = self._extract_error(resp)
            raise APIError(message, status_code=resp.status_code)
        if resp.headers.get("content-type", "").startswith("application/json"):
            return resp.json()
        return resp.content

    @staticmethod
    def _extract_error(resp: requests.Response) -> str:
        try:
            body = resp.json()
            if isinstance(body, dict):
                if "error" in body:
                    return body["error"].get("message", "Request failed")
                if "detail" in body:
                    detail = body["detail"]
                    if isinstance(detail, list) and detail:
                        return detail[0].get("msg", str(detail))
                    return str(detail)
        except ValueError:
            pass
        return f"Request failed ({resp.status_code})"

    # ---- auth ----
    def signup(self, email: str, password: str, full_name: str) -> dict:
        return self._request(
            "POST",
            "/api/auth/signup",
            json={"email": email, "password": password, "full_name": full_name},
        )

    def login(self, email: str, password: str) -> dict:
        return self._request(
            "POST", "/api/auth/login", json={"email": email, "password": password}
        )

    def me(self) -> dict:
        return self._request("GET", "/api/auth/me")

    # ---- chat ----
    def ask(self, message: str, chat_id: str | None, top_k: int | None) -> dict:
        return self._request(
            "POST",
            "/api/chat/ask",
            json={"message": message, "chat_id": chat_id, "top_k": top_k},
        )

    def list_chats(self) -> list[dict]:
        return self._request("GET", "/api/chat/chats")

    def get_chat(self, chat_id: str) -> dict:
        return self._request("GET", f"/api/chat/chats/{chat_id}")

    def delete_chat(self, chat_id: str) -> None:
        self._request("DELETE", f"/api/chat/chats/{chat_id}")

    def feedback(self, message_id: str, rating: str, comment: str | None = None) -> dict:
        return self._request(
            "POST",
            f"/api/chat/messages/{message_id}/feedback",
            json={"rating": rating, "comment": comment},
        )

    # ---- documents ----
    def upload_document(self, filename: str, data: bytes, content_type: str) -> dict:
        return self._request(
            "POST",
            "/api/documents",
            files={"file": (filename, data, content_type)},
        )

    def list_documents(self) -> list[dict]:
        return self._request("GET", "/api/documents")

    def delete_document(self, document_id: str) -> None:
        self._request("DELETE", f"/api/documents/{document_id}")

    # ---- analytics / settings / memory ----
    def analytics_overview(self, days: int = 14) -> dict:
        return self._request("GET", f"/api/analytics/overview?days={days}")

    def public_settings(self) -> dict:
        return self._request("GET", "/api/settings/public")

    def update_profile(self, full_name: str) -> dict:
        return self._request(
            "PUT", "/api/settings/profile", json={"full_name": full_name}
        )

    def get_preferences(self) -> dict:
        return self._request("GET", "/api/memory/preferences")

    def update_preferences(self, prefs: dict) -> dict:
        return self._request("PUT", "/api/memory/preferences", json=prefs)

    # ---- voice ----
    def transcribe(self, filename: str, data: bytes) -> dict:
        return self._request(
            "POST", "/api/voice/transcribe", files={"file": (filename, data)}
        )
