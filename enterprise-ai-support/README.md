# 🛟 Enterprise AI Customer Support Assistant

A production-grade, agentic customer-support platform: multi-document **RAG**
with citations, a **LangGraph** agent supervisor, conversation **memory**,
**voice** I/O, an **analytics** dashboard, **JWT** auth with RBAC, and a premium
dark-glassmorphism **Streamlit** UI over a typed **FastAPI** backend.

> Built to be read by engineers. Clean architecture, type hints, dependency
> injection, structured logging, graceful degradation, and tests throughout.

---

## ✨ Features

| Area | What's included |
|------|-----------------|
| **Document intelligence** | PDF, DOCX, TXT/MD, and image OCR ingestion → semantic chunking → embeddings → vector index |
| **RAG** | Dense retrieval, BM25 hybrid fusion (RRF), re-ranking, context compression, query rewriting, structured citations |
| **Agents** | LangGraph supervisor routing to retrieval / FAQ / memory / analytics specialists + a response synthesiser |
| **Memory** | Conversation history recall + persistent per-user preferences |
| **Auth** | JWT login/signup, bcrypt hashing, role-based access (admin/user), protected routes |
| **Analytics** | Questions, active users, documents, latency, feedback rate — Plotly dashboards |
| **Voice** | Whisper speech-to-text + gTTS text-to-speech |
| **Frontend** | Multi-page Streamlit: Home, Chat, Uploads, Analytics, Settings, Profile |
| **Ops** | Docker + Compose (PostgreSQL), structured logs, request tracing, exception handling, Pytest |

### Graceful degradation
Every heavy dependency has a safe fallback so the app **runs end-to-end with
zero external services**:

- No `LLM_API_KEY` → answers are assembled extractively from retrieved context.
- `sentence-transformers` unavailable → deterministic hashing embedder.
- ChromaDB unavailable → in-process cosine index.
- LangGraph unavailable → equivalent sequential orchestrator.

Configure the real services for full fidelity; the architecture is identical.

---

## 🏗️ Architecture

```
┌──────────────┐     HTTP/JSON      ┌──────────────────────────────────────┐
│  Streamlit   │ ─────────────────▶ │            FastAPI backend            │
│  (frontend)  │ ◀───────────────── │  auth · documents · chat · analytics  │
└──────────────┘                    │  memory · settings · voice            │
                                     └───────────────┬──────────────────────┘
                                                     │
        ┌───────────────────────┬────────────────────┼───────────────────────┐
        ▼                       ▼                     ▼                       ▼
   ┌─────────┐           ┌────────────┐        ┌────────────┐         ┌────────────┐
   │  Auth   │           │    RAG     │        │  Agents    │         │ Analytics  │
   │ JWT/RBAC│           │ load→chunk │        │ LangGraph  │         │ aggregates │
   └────┬────┘           │ →embed→idx │        │ supervisor │         └─────┬──────┘
        │                │ →retrieve  │        │ + routing  │               │
        ▼                │ →rerank    │        └─────┬──────┘               │
   ┌─────────┐           │ →cite      │              │                      │
   │PostgreSQL│◀─────────┴──────┬─────┘              ▼                      │
   │ / SQLite │                 ▼              ┌────────────┐               │
   └─────────┘            ┌──────────┐         │   LLM      │               │
                          │ ChromaDB │         │ (OpenAI-   │◀──────────────┘
                          │ vectors  │         │ compatible)│
                          └──────────┘         └────────────┘
```

Detailed diagrams and design rationale: [`docs/architecture.md`](docs/architecture.md).

---

## 🚀 Quickstart (local, no external services)

```bash
cd enterprise-ai-support
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # optional: add LLM_API_KEY for real answers

# 1) Backend
uvicorn app.main:app --reload          # http://localhost:8000/docs

# 2) Frontend (new terminal)
streamlit run frontend/streamlit_app.py # http://localhost:8501
```

A demo admin is seeded on first startup: **admin@example.com / admin12345**.

### Run with Docker (PostgreSQL + both services)

```bash
export SECRET_KEY=$(python -c "import secrets;print(secrets.token_urlsafe(48))")
export LLM_API_KEY=sk-...        # optional
docker compose up --build
```

---

## 🧪 Tests

```bash
pytest                # unit + integration (uses an isolated temp SQLite DB)
```

The suite covers chunking, embeddings/retrieval/reranking/citations, auth +
JWT, agent routing/orchestration, and end-to-end chat & document APIs.

---

## 📁 Project layout

```
enterprise-ai-support/
├── app/
│   ├── main.py            # FastAPI factory + lifespan (schema, seed admin)
│   ├── config.py          # typed settings (pydantic-settings)
│   ├── database.py        # engine, session, Base, get_db
│   ├── dependencies.py    # DI: db, current user, admin guard
│   ├── exceptions.py      # domain errors + JSON handlers
│   ├── middleware.py      # request-id + access logging
│   ├── models/            # SQLAlchemy ORM (users, chats, messages, docs, …)
│   ├── schemas/           # Pydantic request/response contracts
│   ├── auth/              # hashing, JWT, auth service
│   ├── rag/               # loaders, chunking, embeddings, vector store,
│   │                      #   retriever, reranker, citations, pipeline
│   ├── agents/            # state, tools, nodes, graph (LangGraph + fallback)
│   ├── memory/            # conversation + persistent memory
│   ├── analytics/         # event tracking + dashboard aggregation
│   ├── voice/             # Whisper STT, gTTS TTS
│   ├── services/          # LLM client, chat orchestration
│   └── api/routes/        # auth, documents, chat, analytics, memory,
│                          #   settings, voice, health
├── frontend/              # Streamlit app + pages + lib (api, styles, state)
├── tests/                 # Pytest suite
├── docker/                # backend & frontend Dockerfiles
├── docs/                  # architecture, deployment, API, user, dev guides
├── docker-compose.yml
└── requirements.txt
```

Full responsibilities per folder: [`docs/developer-guide.md`](docs/developer-guide.md).

---

## 📚 Documentation
- [Architecture guide](docs/architecture.md)
- [Deployment guide](docs/deployment.md)
- [API reference](docs/api.md)
- [User guide](docs/user-guide.md)
- [Developer guide](docs/developer-guide.md)

## 📄 License
MIT — see headers. Built as a portfolio reference implementation.
