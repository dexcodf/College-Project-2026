"""Authentication and user schemas."""
from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field

from app.models.user import Role
from app.schemas.common import ORMModel


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(default="", max_length=255)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class UserOut(ORMModel):
    id: str
    email: EmailStr
    full_name: str
    role: Role
    is_active: bool


class UserPreferences(BaseModel):
    theme: str = "dark"
    voice_enabled: bool = False
    default_top_k: int = 8


class TokenWithUser(Token):
    user: UserOut
