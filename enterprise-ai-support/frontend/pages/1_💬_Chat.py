"""Chat workspace — conversation list, streaming-style replies, citations."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from frontend.lib.api import APIError  # noqa: E402
from frontend.lib.state import get_client, require_auth  # noqa: E402
from frontend.lib.styles import inject_css  # noqa: E402

st.set_page_config(page_title="Chat · AI Support", page_icon="💬", layout="wide")
inject_css()
require_auth()

client = get_client()


def _render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander(f"📎 Sources ({len(citations)})"):
        for i, c in enumerate(citations, start=1):
            page = f" · p.{c['page']}" if c.get("page") else ""
            st.markdown(
                f'<div class="citation"><b>[{i}] {c["filename"]}{page}</b> '
                f'· score {c["score"]:.2f}<br>{c["snippet"]}</div>',
                unsafe_allow_html=True,
            )


# ---- sidebar: conversation list ----
with st.sidebar:
    st.markdown("### 💬 Conversations")
    if st.button("➕ New chat", use_container_width=True):
        st.session_state["active_chat_id"] = None
        st.session_state["messages"] = []
        st.rerun()
    try:
        chats = client.list_chats()
    except APIError as exc:
        st.error(str(exc))
        chats = []
    for chat in chats:
        if st.button(
            chat["title"][:34] or "Untitled", key=f"chat_{chat['id']}", use_container_width=True
        ):
            st.session_state["active_chat_id"] = chat["id"]
            detail = client.get_chat(chat["id"])
            st.session_state["messages"] = detail["messages"]
            st.rerun()

# ---- main: message history ----
st.markdown("#### Ask your knowledge base")
messages = st.session_state.get("messages", [])
for msg in messages:
    with st.chat_message("assistant" if msg["role"] == "assistant" else "user"):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            _render_citations(msg.get("citations", []))
            if msg.get("agent_route"):
                st.caption(f"routed via **{msg['agent_route']}** agent")

# ---- input ----
prompt = st.chat_input("Type your question…")
if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                res = client.ask(
                    prompt, st.session_state.get("active_chat_id"), top_k=None
                )
                st.session_state["active_chat_id"] = res["chat_id"]
                answer = res["message"]
                st.write(answer["content"])
                _render_citations(answer.get("citations", []))

                # Feedback buttons.
                col_up, col_down, _ = st.columns([1, 1, 6])
                with col_up:
                    if st.button("👍", key=f"up_{answer['id']}"):
                        client.feedback(answer["id"], "up")
                        st.toast("Thanks for the feedback!")
                with col_down:
                    if st.button("👎", key=f"down_{answer['id']}"):
                        client.feedback(answer["id"], "down")
                        st.toast("Feedback recorded.")

                # Persist into local history.
                st.session_state.setdefault("messages", [])
                st.session_state["messages"].append(
                    {"role": "user", "content": prompt, "citations": []}
                )
                st.session_state["messages"].append(answer)
            except APIError as exc:
                st.error(str(exc))
