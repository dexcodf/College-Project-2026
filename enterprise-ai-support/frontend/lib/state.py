"""Session-state helpers shared across pages."""
from __future__ import annotations

import streamlit as st

from frontend.lib.api import APIClient


def get_client() -> APIClient:
    """Return an APIClient bound to the logged-in user's token."""
    return APIClient(token=st.session_state.get("token"))


def is_authenticated() -> bool:
    return bool(st.session_state.get("token"))


def require_auth() -> None:
    """Stop rendering a page if the user is not logged in."""
    if not is_authenticated():
        st.warning("Please sign in from the Home page to continue.")
        st.stop()


def current_user() -> dict:
    return st.session_state.get("user", {})


def login(token: str, user: dict) -> None:
    st.session_state["token"] = token
    st.session_state["user"] = user


def logout() -> None:
    for key in ("token", "user", "active_chat_id", "messages"):
        st.session_state.pop(key, None)
