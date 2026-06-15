# Presentation & Demo Guide

A 5-minute script to demo the **Enterprise AI Customer Support Assistant**.

## Before the demo
```bash
# 1. Start everything (Windows)
powershell -ExecutionPolicy Bypass -File run.ps1
#    …or macOS/Linux
./run.sh

# 2. Seed realistic demo data (one time, while servers are up)
python scripts/seed_demo.py
```
Open **http://127.0.0.1:8501** and sign in: `admin@example.com` / `admin12345`.

## The 90-second elevator pitch
> "This is an enterprise customer-support assistant — think Glean or Intercom's
> AI. You upload your company's documents; it answers customer questions grounded
> in those documents, **with citations**, so answers are trustworthy and
> auditable. Under the hood it's a Retrieval-Augmented Generation pipeline driven
> by a multi-agent LangGraph workflow, with auth, memory, analytics, and voice —
> all production-structured with a FastAPI backend and a Streamlit UI."

## Demo flow (what to click)
1. **Home** — point out the dark, glassmorphic SaaS UI and the capability badges.
2. **📁 Uploads** — show the seeded knowledge base (shipping, returns, warranty,
   billing). Optionally drag in a new PDF live to show ingestion + chunk count.
3. **💬 Chat** — ask: *"What's your return window and how are refunds handled?"*
   - Expand **📎 Sources** → show file + page + relevance score + snippet.
   - Note the caption: which **agent** answered (retrieval vs FAQ).
   - Click 👍 to show feedback capture.
   - Ask a FAQ: *"How do I reset my password?"* → routed to the **FAQ agent**.
4. **📈 Analytics** — questions over time, response latency, agent-routing
   distribution, feedback rate. (Populated by the seed script.)
5. **⚙️ Settings / 👤 Profile** — preferences, runtime config, RBAC role.
6. **API** — open **http://127.0.0.1:8000/docs** to show the typed REST API.

## Talking points (the "why it's good" slide)
- **Grounded + cited** answers — not hallucinated; every claim links to a source.
- **Hybrid retrieval** — dense (BGE embeddings) + BM25 fused with Reciprocal Rank
  Fusion, then re-ranked and compressed.
- **Agentic** — a supervisor routes each query to the right specialist agent.
- **Production engineering** — JWT/RBAC auth, structured logging with request
  tracing, dependency injection, typed exceptions, 26 passing tests, Docker
  Compose with PostgreSQL.
- **Provider-flexible** — one config switch targets OpenAI, Groq, Ollama, or vLLM.
- **Resilient** — graceful fallbacks mean it runs even with no GPU/API key.

## If asked "is it really working?"
- `GET /api/health` reports live capabilities (LLM, embeddings, agent backend).
- `pytest` → 26 passing tests covering auth, RAG, agents, and the API.

## FAQ from reviewers
- **"Where do vectors live?"** ChromaDB; chunk metadata stays in SQL so citations
  are relational and per-user filtering is cheap.
- **"How would you scale it?"** Backend is stateless; move Chroma to a shared
  server and embedding/Whisper inference to a GPU worker — the interfaces are the
  seams. See `docs/deployment.md`.
- **"Can it use a different model?"** Yes — set `LLM_BASE_URL` + `LLM_MODEL`.
