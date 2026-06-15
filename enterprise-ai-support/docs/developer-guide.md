# Developer Guide

## Local setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload          # backend
streamlit run frontend/streamlit_app.py # frontend (separate terminal)
pytest                                  # tests
```

Python 3.11+. The first run downloads the embedding model (and Whisper if you
use voice); without network access the fallbacks engage automatically.

## Folder responsibilities

| Path | Responsibility |
|------|----------------|
| `app/config.py` | Typed settings; the only place env vars are read. |
| `app/database.py` | Engine, `SessionLocal`, `Base`, `get_db`, `init_db`. |
| `app/dependencies.py` | DI providers: `DbSession`, `CurrentUser`, `AdminUser`. |
| `app/exceptions.py` | `AppError` hierarchy + JSON exception handlers. |
| `app/middleware.py` | Request-id binding + access/latency logging. |
| `app/models/*` | SQLAlchemy ORM; `__init__` registers all tables. |
| `app/schemas/*` | Pydantic request/response models (the API contract). |
| `app/auth/security.py` | bcrypt hashing + JWT encode/decode. |
| `app/auth/service.py` | Signup/authenticate/lookup; admin bootstrap. |
| `app/rag/loaders.py` | Per-format extraction (incl. OCR) → `LoadedPage`. |
| `app/rag/chunking.py` | Recursive, page-aware, overlapping chunking. |
| `app/rag/embeddings.py` | BGE embeddings + hashing fallback. |
| `app/rag/vector_store.py` | ChromaDB wrapper + in-memory cosine fallback. |
| `app/rag/retriever.py` | Dense → BM25 RRF → rerank → compress. |
| `app/rag/reranker.py` | Lexical/dense blended re-ranking. |
| `app/rag/citations.py` | Chunks → `Citation` objects + prompt context. |
| `app/rag/pipeline.py` | Ingest write-path; status transitions. |
| `app/agents/nodes.py` | Pure node functions (supervisor/specialists/response). |
| `app/agents/graph.py` | LangGraph build + sequential fallback + `AgentRunner`. |
| `app/agents/tools.py` | KB search, FAQ lookup, KB size. |
| `app/memory/store.py` | Conversation recall + persistent preferences. |
| `app/analytics/service.py` | Event tracking + dashboard aggregation. |
| `app/voice/*` | Whisper STT, gTTS TTS. |
| `app/services/llm.py` | OpenAI-compatible client + offline fallback. |
| `app/services/chat_service.py` | Full Q&A turn orchestration. |
| `app/api/routes/*` | One router per domain; aggregated in `__init__`. |
| `frontend/lib/*` | API client, CSS, session-state helpers. |
| `frontend/pages/*` | Streamlit pages (Chat, Uploads, Analytics, …). |

## Conventions
- **Type hints everywhere**; `from __future__ import annotations` at the top.
- Routes are thin; logic lives in services. Services don't import FastAPI.
- Raise typed `AppError` subclasses; handlers map them to HTTP.
- Log with `get_logger(name)`; bind context, never `print`.
- New optional dependency? Add a fallback so the app stays runnable.

## Extending

| To add… | Do this |
|---------|---------|
| A new document format | Add a loader in `rag/loaders.py` + extension. |
| A new agent | Add a node in `agents/nodes.py`, register in `graph.py`'s `_SPECIALISTS`, add a route branch in `supervisor_node`. |
| A new API resource | New router in `api/routes/`, include it in `routes/__init__.py`. |
| LLM-based routing/rewriting | Swap the heuristic in `supervisor_node` / `rewrite_query` for an `llm_client` call — interfaces are unchanged. |
| A real cross-encoder | Replace the body of `reranker.rerank`. |

## Testing
- `tests/conftest.py` spins up an isolated temp SQLite DB and overrides
  `get_db`; `auth_headers` registers a fresh user per use.
- Pure modules (chunking, reranking, citations) are tested without I/O.
- API tests exercise the real app via `TestClient` end-to-end.
```bash
pytest -q                  # all
pytest tests/test_rag.py   # one file
```
