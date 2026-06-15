"""User profile."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from frontend.lib.api import APIError  # noqa: E402
from frontend.lib.state import current_user, get_client, require_auth  # noqa: E402
from frontend.lib.styles import inject_css  # noqa: E402

st.set_page_config(page_title="Profile · AI Support", page_icon="👤", layout="wide")
inject_css()
require_auth()

client = get_client()
user = current_user()

st.markdown('<div class="hero-title">Profile</div>', unsafe_allow_html=True)

st.markdown(
    f'<div class="glass">'
    f'<p><b>Email:</b> {user.get("email")}</p>'
    f'<p><b>Role:</b> <span class="badge ok">{user.get("role")}</span></p>'
    f'<p><b>Account ID:</b> <code>{user.get("id")}</code></p>'
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown("#### Update display name")
with st.form("profile_form"):
    name = st.text_input("Full name", value=user.get("full_name", ""))
    if st.form_submit_button("Save"):
        try:
            updated = client.update_profile(name)
            st.session_state["user"] = updated
            st.success("Profile updated.")
        except APIError as exc:
            st.error(str(exc))
