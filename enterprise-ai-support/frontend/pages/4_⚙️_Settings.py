"""Settings — preferences and runtime configuration."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from frontend.lib.api import APIError  # noqa: E402
from frontend.lib.state import get_client, require_auth  # noqa: E402
from frontend.lib.styles import inject_css  # noqa: E402

st.set_page_config(page_title="Settings · AI Support", page_icon="⚙️", layout="wide")
inject_css()
require_auth()

client = get_client()

st.markdown('<div class="hero-title">Settings</div>', unsafe_allow_html=True)

st.markdown("#### Preferences")
try:
    prefs = client.get_preferences()
except APIError as exc:
    st.error(str(exc))
    prefs = {"theme": "dark", "voice_enabled": False, "default_top_k": 8}

with st.form("prefs_form"):
    theme = st.selectbox(
        "Theme", ["dark", "light"], index=0 if prefs.get("theme") == "dark" else 1
    )
    voice = st.toggle("Enable voice input", value=prefs.get("voice_enabled", False))
    top_k = st.slider("Default retrieval depth (top_k)", 1, 20, prefs.get("default_top_k", 8))
    if st.form_submit_button("Save preferences"):
        try:
            client.update_preferences(
                {"theme": theme, "voice_enabled": voice, "default_top_k": top_k}
            )
            st.success("Preferences saved.")
        except APIError as exc:
            st.error(str(exc))

st.divider()
st.markdown("#### Runtime configuration")
try:
    info = client.public_settings()
    st.json(info)
except APIError as exc:
    st.error(str(exc))
