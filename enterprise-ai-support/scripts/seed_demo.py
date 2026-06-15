"""Seed the running app with realistic demo data for presentations.

Drives the real HTTP API (no DB shortcuts) so the knowledge base, chats, and
analytics dashboard are all populated through the genuine pipeline.

Usage:
    python scripts/seed_demo.py                 # uses http://127.0.0.1:8000
    BACKEND_URL=http://host:8000 python scripts/seed_demo.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
ADMIN_EMAIL = os.getenv("SEED_ADMIN_EMAIL", "admin@example.com")
ADMIN_PASSWORD = os.getenv("SEED_ADMIN_PASSWORD", "admin12345")

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_docs"

# Questions chosen to exercise every agent route (retrieval / faq / analytics).
DEMO_QUESTIONS = [
    "How long does standard shipping take and is it free?",
    "What is your return window and how are refunds processed?",
    "How long is the product warranty and what does it cover?",
    "How do I reset my password?",
    "How many documents are indexed in the knowledge base?",
    "Can I cancel my subscription before renewal?",
]


def _die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    s = requests.Session()

    # 1) Authenticate as the seeded admin.
    try:
        r = s.post(
            f"{BACKEND_URL}/api/auth/login",
            json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD},
            timeout=30,
        )
    except requests.RequestException as exc:
        _die(f"cannot reach backend at {BACKEND_URL}: {exc}")
    if r.status_code != 200:
        _die(f"admin login failed ({r.status_code}): {r.text}")
    token = r.json()["access_token"]
    s.headers["Authorization"] = f"Bearer {token}"
    print(f"✓ authenticated as {ADMIN_EMAIL}")

    # 2) Upload the sample knowledge base.
    existing = {d["filename"] for d in s.get(f"{BACKEND_URL}/api/documents").json()}
    docs = sorted(SAMPLE_DIR.glob("*.md"))
    if not docs:
        _die(f"no sample documents found in {SAMPLE_DIR}")
    for doc in docs:
        if doc.name in existing:
            print(f"• {doc.name} already indexed — skipping")
            continue
        r = s.post(
            f"{BACKEND_URL}/api/documents",
            files={"file": (doc.name, doc.read_bytes(), "text/markdown")},
            timeout=120,
        )
        if r.status_code == 201:
            body = r.json()
            print(f"✓ ingested {doc.name}: {body['chunks_created']} chunks")
        else:
            print(f"✗ {doc.name}: {r.status_code} {r.text}")

    # 3) Ask demo questions to populate chats + analytics.
    for q in DEMO_QUESTIONS:
        r = s.post(
            f"{BACKEND_URL}/api/chat/ask", json={"message": q}, timeout=120
        )
        if r.status_code != 200:
            print(f"✗ ask failed: {r.status_code} {r.text}")
            continue
        msg = r.json()["message"]
        print(
            f"✓ asked: {q[:48]:50s} → route={msg['agent_route']:9s} "
            f"cites={len(msg['citations'])}"
        )
        # Leave a thumbs-up so the feedback rate is non-trivial.
        s.post(
            f"{BACKEND_URL}/api/chat/messages/{msg['id']}/feedback",
            json={"rating": "up"},
            timeout=30,
        )

    print("\n🎉 Demo data seeded. Open the Analytics page to see it populated.")


if __name__ == "__main__":
    main()
