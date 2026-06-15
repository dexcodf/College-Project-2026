"""Analytics dashboard — Plotly visualisations (admin only)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from frontend.lib.api import APIError  # noqa: E402
from frontend.lib.state import current_user, get_client, require_auth  # noqa: E402
from frontend.lib.styles import inject_css, metric_card  # noqa: E402

st.set_page_config(page_title="Analytics · AI Support", page_icon="📈", layout="wide")
inject_css()
require_auth()

client = get_client()

st.markdown('<div class="hero-title">Analytics</div>', unsafe_allow_html=True)

if current_user().get("role") != "admin":
    st.warning("Analytics is restricted to administrators.")
    st.stop()

days = st.slider("Window (days)", min_value=7, max_value=60, value=14)
try:
    data = client.analytics_overview(days=days)
except APIError as exc:
    st.error(str(exc))
    st.stop()

# ---- KPI row ----
cols = st.columns(5)
kpis = [
    ("Questions", f"{data['total_questions']:,}"),
    ("Users", f"{data['total_users']:,}"),
    ("Active (7d)", f"{data['active_users_7d']:,}"),
    ("Documents", f"{data['total_documents']:,}"),
    ("Avg latency", f"{data['avg_response_ms']:.0f} ms"),
]
for col, (label, value) in zip(cols, kpis):
    col.markdown(metric_card(label, value), unsafe_allow_html=True)

st.markdown(
    f"**Positive feedback rate:** {data['positive_feedback_rate'] * 100:.1f}%"
)
st.divider()


def _line(points: list[dict], name: str, color: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[p["date"] for p in points],
            y=[p["value"] for p in points],
            mode="lines+markers",
            name=name,
            line={"color": color, "width": 3},
            fill="tozeroy",
            fillcolor=color.replace("1)", "0.15)"),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        height=300,
        title=name,
    )
    return fig


c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(
        _line(data["questions_over_time"], "Questions over time", "rgba(99,102,241,1)"),
        use_container_width=True,
    )
with c2:
    st.plotly_chart(
        _line(
            data["response_times_over_time"],
            "Avg response time (ms)",
            "rgba(34,211,238,1)",
        ),
        use_container_width=True,
    )

# ---- routing distribution ----
routes = data.get("top_routes", {})
if routes:
    fig = go.Figure(
        go.Bar(
            x=list(routes.values()),
            y=list(routes.keys()),
            orientation="h",
            marker={"color": "rgba(99,102,241,0.85)"},
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        title="Agent routing distribution",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
