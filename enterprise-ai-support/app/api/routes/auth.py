"""Authentication endpoints: signup, login, current user."""
from __future__ import annotations

from fastapi import APIRouter, status

from app.auth.security import create_access_token
from app.dependencies import AuthServiceDep, CurrentUser
from app.schemas.auth import (
    LoginRequest,
    SignupRequest,
    TokenWithUser,
    UserOut,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=TokenWithUser, status_code=status.HTTP_201_CREATED)
def signup(payload: SignupRequest, auth: AuthServiceDep) -> TokenWithUser:
    user = auth.signup(payload.email, payload.password, payload.full_name)
    token, expires_in = create_access_token(user.id, role=user.role.value)
    return TokenWithUser(
        access_token=token, expires_in=expires_in, user=UserOut.model_validate(user)
    )


@router.post("/login", response_model=TokenWithUser)
def login(payload: LoginRequest, auth: AuthServiceDep) -> TokenWithUser:
    user = auth.authenticate(payload.email, payload.password)
    token, expires_in = create_access_token(user.id, role=user.role.value)
    return TokenWithUser(
        access_token=token, expires_in=expires_in, user=UserOut.model_validate(user)
    )


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser) -> UserOut:
    return UserOut.model_validate(user)
