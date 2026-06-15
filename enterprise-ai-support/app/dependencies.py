"""FastAPI dependency-injection providers.

Centralises the wiring between HTTP requests and services: DB sessions,
the authenticated user, and role-based guards.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.auth.security import decode_token
from app.auth.service import AuthService
from app.database import get_db
from app.exceptions import AuthError, PermissionError_
from app.models.user import Role, User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

DbSession = Annotated[Session, Depends(get_db)]


def get_auth_service(db: DbSession) -> AuthService:
    return AuthService(db)


AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]


def get_current_user(
    token: Annotated[str | None, Depends(oauth2_scheme)],
    auth: AuthServiceDep,
) -> User:
    """Resolve the authenticated user from a bearer token."""
    if not token:
        raise AuthError("Not authenticated")
    payload = decode_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise AuthError("Invalid token payload")
    user = auth.get_by_id(user_id)
    if not user.is_active:
        raise AuthError("Account is disabled")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_admin(user: CurrentUser) -> User:
    """Guard that allows only admins through."""
    if user.role != Role.ADMIN:
        raise PermissionError_("Administrator privileges required")
    return user


AdminUser = Annotated[User, Depends(require_admin)]
