"""Cryptographic primitives: password hashing and JWT encode/decode."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from jose import JWTError, jwt

from app.config import settings
from app.exceptions import AuthError

# bcrypt operates on at most 72 bytes; longer inputs are truncated (standard
# behaviour). We use the bcrypt library directly to avoid passlib's shim, which
# is incompatible with bcrypt >= 4.1.
_BCRYPT_MAX_BYTES = 72


def _encode(password: str) -> bytes:
    return password.encode("utf-8")[:_BCRYPT_MAX_BYTES]


def hash_password(password: str) -> str:
    """Return a bcrypt hash of the plaintext password."""
    return bcrypt.hashpw(_encode(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time verification of a plaintext password against a hash."""
    try:
        return bcrypt.checkpw(_encode(plain), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(
    subject: str, *, role: str, expires_minutes: int | None = None
) -> tuple[str, int]:
    """Create a signed JWT. Returns (token, expires_in_seconds)."""
    minutes = expires_minutes or settings.access_token_expire_minutes
    expire = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    payload: dict[str, Any] = {
        "sub": subject,
        "role": role,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    token = jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)
    return token, minutes * 60


def decode_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT, raising AuthError on any failure."""
    try:
        return jwt.decode(
            token, settings.secret_key, algorithms=[settings.algorithm]
        )
    except JWTError as exc:  # expired, bad signature, malformed, ...
        raise AuthError("Invalid or expired token") from exc
