"""Auth service: signup, authentication, and user lookup."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.security import hash_password, verify_password
from app.exceptions import AuthError, ConflictError, NotFoundError
from app.logging_config import get_logger
from app.models.user import Role, User

logger = get_logger("auth")


class AuthService:
    """Encapsulates user account operations against the database."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_by_email(self, email: str) -> User | None:
        return self.db.scalar(select(User).where(User.email == email.lower()))

    def get_by_id(self, user_id: str) -> User:
        user = self.db.get(User, user_id)
        if user is None:
            raise NotFoundError("User not found")
        return user

    def signup(
        self, email: str, password: str, full_name: str = "", role: Role = Role.USER
    ) -> User:
        email = email.lower()
        if self.get_by_email(email) is not None:
            raise ConflictError("An account with this email already exists")
        user = User(
            email=email,
            full_name=full_name,
            hashed_password=hash_password(password),
            role=role,
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        logger.info("user_signed_up", user_id=user.id, role=role.value)
        return user

    def authenticate(self, email: str, password: str) -> User:
        user = self.get_by_email(email)
        if user is None or not verify_password(password, user.hashed_password):
            # Same error for both cases to avoid user enumeration.
            raise AuthError("Incorrect email or password")
        if not user.is_active:
            raise AuthError("Account is disabled")
        logger.info("user_authenticated", user_id=user.id)
        return user

    def ensure_default_admin(
        self, email: str, password: str, full_name: str = "Administrator"
    ) -> User:
        """Idempotently create a bootstrap admin (used at startup)."""
        existing = self.get_by_email(email)
        if existing is not None:
            return existing
        return self.signup(email, password, full_name, role=Role.ADMIN)
