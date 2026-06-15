"""Premium dark theme + glassmorphism CSS injected into every page."""
from __future__ import annotations

import streamlit as st

_CSS = """
<style>
:root {
  --bg-0: #0a0e1a;
  --bg-1: #111729;
  --glass: rgba(255, 255, 255, 0.05);
  --glass-border: rgba(255, 255, 255, 0.12);
  --accent: #6366f1;
  --accent-2: #22d3ee;
  --text: #e8eaf2;
  --text-dim: #9aa3b8;
}

.stApp {
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(99,102,241,0.18), transparent),
    radial-gradient(1000px 500px at 110% 10%, rgba(34,211,238,0.12), transparent),
    var(--bg-0);
  color: var(--text);
}

/* Glass cards */
.glass {
  background: var(--glass);
  border: 1px solid var(--glass-border);
  border-radius: 18px;
  padding: 1.25rem 1.4rem;
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}

.hero-title {
  font-size: 2.4rem; font-weight: 800; letter-spacing: -0.02em;
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin-bottom: 0.2rem;
}
.hero-sub { color: var(--text-dim); font-size: 1.05rem; margin-bottom: 1.4rem; }

/* Metric cards */
.metric {
  background: var(--glass); border: 1px solid var(--glass-border);
  border-radius: 16px; padding: 1.1rem 1.2rem; backdrop-filter: blur(12px);
}
.metric .label { color: var(--text-dim); font-size: 0.82rem; text-transform: uppercase;
  letter-spacing: 0.06em; }
.metric .value { font-size: 1.9rem; font-weight: 750; margin-top: 0.2rem; }

/* Citation chips */
.citation {
  display:block; background: rgba(99,102,241,0.10); border:1px solid var(--glass-border);
  border-left: 3px solid var(--accent); border-radius: 10px; padding: 0.6rem 0.8rem;
  margin: 0.4rem 0; font-size: 0.86rem; color: var(--text-dim);
}
.citation b { color: var(--text); }

/* Buttons */
.stButton > button {
  border-radius: 12px; border: 1px solid var(--glass-border);
  background: linear-gradient(90deg, rgba(99,102,241,0.9), rgba(34,211,238,0.85));
  color: white; font-weight: 600; transition: transform .08s ease, box-shadow .2s ease;
}
.stButton > button:hover { transform: translateY(-1px);
  box-shadow: 0 6px 20px rgba(99,102,241,0.35); }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--bg-1), var(--bg-0));
  border-right: 1px solid var(--glass-border);
}

/* Chat bubbles via st.chat_message tweaks */
[data-testid="stChatMessage"] {
  background: var(--glass); border: 1px solid var(--glass-border);
  border-radius: 16px; backdrop-filter: blur(10px);
}

.badge { display:inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
  font-size: 0.72rem; font-weight:600; border:1px solid var(--glass-border);
  background: rgba(255,255,255,0.06); color: var(--text-dim); }
.badge.ok { color:#34d399; border-color: rgba(52,211,153,0.4); }
.badge.warn { color:#fbbf24; border-color: rgba(251,191,36,0.4); }
</style>
"""


def inject_css() -> None:
    """Apply the global theme. Call once near the top of every page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def metric_card(label: str, value: str) -> str:
    return f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div></div>'
