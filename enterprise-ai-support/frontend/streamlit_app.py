"""Home / authentication page — entrypoint for the Streamlit app.

Run from the project root:
    streamlit run frontend/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable so `frontend.lib.*` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from frontend.lib.api import APIClient, APIError  # noqa: E402
from frontend.lib.state import current_user, is_authenticated, login, logout  # noqa: E402
from frontend.lib.styles import inject_css  # noqa: E402

st.set_page_config(
    page_title="Enterprise AI Support",
    page_icon="🛟",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


def _auth_panel() -> None:
    st.markdown('<div class="hero-title">Enterprise AI Support</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Agentic, document-grounded customer support — '
        "RAG, citations, memory, analytics, and voice.</div>",
        unsafe_allow_html=True,
    )

    tab_login, tab_signup = st.tabs(["Sign in", "Create account"])
    client = APIClient()

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email", value="admin@example.com")
            password = st.text_input("Password", type="password", value="admin12345")
            submitted = st.form_submit_button("Sign in", use_container_width=True)
        if submitted:
            try:
                res = client.login(email, password)
                login(res["access_token"], res["user"])
                st.success("Signed in. Use the sidebar to navigate.")
                st.rerun()
            except APIError as exc:
                st.error(str(exc))
        st.caption("Demo admin is seeded automatically on first backend startup.")

    with tab_signup:
        with st.form("signup_form"):
            name = st.text_input("Full name")
            email = st.text_input("Email", key="su_email")
            password = st.text_input(
                "Password (min 8 chars)", type="password", key="su_pw"
            )
            submitted = st.form_submit_button("Create account", use_container_width=True)
        if submitted:
            try:
                res = client.signup(email, password, name)
                login(res["access_token"], res["user"])
                st.success("Account created.")
                st.rerun()
            except APIError as exc:
                st.error(str(exc))


def _dashboard_home() -> None:
    user = current_user()
    st.markdown(
        f'<div class="hero-title">Welcome back, {user.get("full_name") or user.get("email")}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-sub">Choose a workspace from the sidebar.</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    cards = [
        ("💬 Chat", "Ask grounded questions with citations."),
        ("📁 Uploads", "Add PDFs, DOCX, TXT, or images (OCR)."),
        ("📈 Analytics", "Usage, latency, and feedback metrics."),
    ]
    for col, (title, desc) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="glass"><h3>{title}</h3><p style="color:var(--text-dim)">{desc}</p></div>',
                unsafe_allow_html=True,
            )

    st.divider()
    try:
        info = APIClient().public_settings()
        st.markdown(
            f'<span class="badge ok">LLM: {info["llm_model"]}</span> '
            f'<span class="badge">Embeddings: {info["embedding_model"]}</span> '
            f'<span class="badge">Env: {info["environment"]}</span>',
            unsafe_allow_html=True,
        )
    except APIError:
        st.markdown('<span class="badge warn">Backend offline</span>', unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### 🛟 AI Support")
    if is_authenticated():
        st.caption(current_user().get("email", ""))
        if st.button("Log out", use_container_width=True):
            logout()
            st.rerun()
    else:
        st.caption("Not signed in")

if is_authenticated():
    _dashboard_home()
else:
    _auth_panel()
