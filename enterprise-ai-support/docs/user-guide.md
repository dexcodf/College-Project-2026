# User Guide

## Getting started
1. Open the app (default `http://localhost:8501`).
2. **Sign in** with the demo admin (`admin@example.com` / `admin12345`) or
   **Create account**.
3. Use the left sidebar to move between workspaces.

## Uploading documents (📁 Uploads)
- Drag in **PDF, DOCX, TXT, MD, or images** (PNG/JPG — text is read via OCR).
- Click **Ingest**. Each file is chunked, embedded, and indexed; you'll see the
  chunk count and a `ready` badge.
- Delete a document with 🗑 to also remove its vectors.

## Chatting (💬 Chat)
- Type a question in the input bar. Answers are **grounded in your documents**.
- Expand **📎 Sources** to see citations (file, page, relevance score, snippet).
- The caption shows which **agent** handled the query (retrieval/FAQ/…).
- Rate answers with 👍 / 👎 — this feeds the analytics feedback rate.
- **➕ New chat** starts a fresh conversation; past chats are in the sidebar.

> Without an `LLM_API_KEY` configured, answers are assembled directly from the
> most relevant retrieved passages (still cited). Configure a key for fully
> synthesised, conversational answers.

## Analytics (📈 Analytics, admin only)
KPIs (questions, users, active users, documents, latency), questions and
response-time trends, and the agent routing distribution. Adjust the day window
with the slider.

## Settings (⚙️) & Profile (👤)
- **Settings**: theme, voice toggle, default retrieval depth; view runtime
  config (model names, environment).
- **Profile**: see your account details and update your display name.

## Tips
- Ask specific questions; the retriever rewards precise wording.
- Upload focused documents (policies, manuals, FAQs) for the best citations.
- Common questions (password reset, refunds, hours, contact) are answered
  instantly by the FAQ agent.
