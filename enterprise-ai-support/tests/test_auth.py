"""Auth: hashing, JWT round-trip, and API flows."""
from __future__ import annotations

import pytest

from app.auth.security import (
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.exceptions import AuthError


def test_password_hash_roundtrip():
    hashed = hash_password("s3cret-pass")
    assert hashed != "s3cret-pass"
    assert verify_password("s3cret-pass", hashed)
    assert not verify_password("wrong", hashed)


def test_jwt_roundtrip():
    token, expires_in = create_access_token("user-123", role="user")
    assert expires_in > 0
    payload = decode_token(token)
    assert payload["sub"] == "user-123"
    assert payload["role"] == "user"


def test_decode_invalid_token_raises():
    with pytest.raises(AuthError):
        decode_token("not.a.jwt")


def test_signup_and_login_flow(client):
    email = "flow@example.com"
    r1 = client.post(
        "/api/auth/signup",
        json={"email": email, "password": "password123", "full_name": "Flow"},
    )
    assert r1.status_code == 201
    assert r1.json()["user"]["email"] == email

    r2 = client.post(
        "/api/auth/login", json={"email": email, "password": "password123"}
    )
    assert r2.status_code == 200
    token = r2.json()["access_token"]

    r3 = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r3.status_code == 200
    assert r3.json()["email"] == email


def test_duplicate_signup_conflicts(client):
    payload = {"email": "dup@example.com", "password": "password123", "full_name": "D"}
    assert client.post("/api/auth/signup", json=payload).status_code == 201
    assert client.post("/api/auth/signup", json=payload).status_code == 409


def test_protected_route_requires_token(client):
    assert client.get("/api/auth/me").status_code == 401
