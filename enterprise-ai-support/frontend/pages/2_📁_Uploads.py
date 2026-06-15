"""Upload center — ingest documents into the knowledge base."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from frontend.lib.api import APIError  # noqa: E402
from frontend.lib.state import get_client, require_auth  # noqa: E402
from frontend.lib.styles import inject_css  # noqa: E402

st.set_page_config(page_title="Uploads · AI Support", page_icon="📁", layout="wide")
inject_css()
require_auth()

client = get_client()

st.markdown('<div class="hero-title">Upload Center</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">PDF, DOCX, TXT, MD, or images (OCR). Files are chunked, '
    "embedded, and indexed for retrieval.</div>",
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Drop a file",
    type=["pdf", "docx", "txt", "md", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)
if uploaded and st.button("Ingest", use_container_width=False):
    for f in uploaded:
        with st.spinner(f"Ingesting {f.name}…"):
            try:
                res = client.upload_document(
                    f.name, f.getvalue(), f.type or "application/octet-stream"
                )
                st.success(
                    f"✅ {f.name}: {res['chunks_created']} chunks "
                    f"({res['document']['status']})"
                )
            except APIError as exc:
                st.error(f"❌ {f.name}: {exc}")

st.divider()
st.markdown("#### Your documents")
try:
    docs = client.list_documents()
except APIError as exc:
    st.error(str(exc))
    docs = []

if not docs:
    st.info("No documents yet. Upload one above to start building your knowledge base.")
for doc in docs:
    col_a, col_b, col_c, col_d = st.columns([4, 2, 2, 1])
    col_a.write(f"📄 **{doc['filename']}**")
    badge = "ok" if doc["status"] == "ready" else "warn"
    col_b.markdown(
        f'<span class="badge {badge}">{doc["status"]}</span>', unsafe_allow_html=True
    )
    col_c.caption(f"{doc['chunk_count']} chunks · {doc['size_bytes'] // 1024} KB")
    if col_d.button("🗑", key=f"del_{doc['id']}"):
        client.delete_document(doc["id"])
        st.rerun()
    if doc.get("error"):
        st.caption(f"⚠️ {doc['error']}")
