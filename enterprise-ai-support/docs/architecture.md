# Architecture Guide

## 1. Goals & principles

- **Clean layering**: HTTP → services → domain → infrastructure. Routes never
  contain business logic; services never import FastAPI.
- **Dependency injection**: DB sessions, the current user, and role guards are
  provided via FastAPI `Depends` (`app/dependencies.py`), making everything
  unit-testable with overrides.
- **Single source of config**: `app/config.py` validates all settings once.
- **Graceful degradation**: optional heavy dependencies (LLM, embeddings,
  ChromaDB, LangGraph) each have a runnable fallback so the system never
  hard-fails in CI/demo.
- **Observability**: structured logs (`structlog`) with a per-request id.

## 2. High-level components

| Layer | Module(s) | Responsibility |
|-------|-----------|----------------|
| API | `app/api/routes/*` | HTTP contracts, validation, status codes |
| Services | `app/services/*` | Orchestration (chat turn, LLM access) |
| Auth | `app/auth/*` | Hashing, JWT, account management |
| RAG | `app/rag/*` | Ingest + retrieval pipelines |
| Agents | `app/agents/*` | Routing + answer synthesis (LangGraph) |
| Memory | `app/memory/*` | Conversation recall + user preferences |
| Analytics | `app/analytics/*` | Event log + dashboard aggregation |
| Persistence | `app/models/*`, `database.py` | ORM + sessions |

## 3. Request → answer data flow (chat)

1. `POST /api/chat/ask` → `chat.py` route resolves `CurrentUser` via JWT.
2. `ChatService.ask` creates/loads the `Chat`, persists the user `Message`.
3. `ConversationMemory.recent_turns` builds prompt history within a budget.
4. `AgentRunner.run(state)` executes the graph:
   - **supervisor** routes (FAQ / memory / analytics / retrieval).
   - **retrieval** node runs the RAG pipeline → chunks, context, citations.
   - **response** node calls the LLM grounded in context + history.
5. The assistant `Message` is persisted with citations, latency, and route.
6. `AnalyticsService.track("question_asked", value=latency)` records metrics.

## 4. RAG pipeline

**Ingest** (`rag/pipeline.py`): save file → `loaders.load_document` (PDF/DOCX/
TXT/image-OCR) → `chunking.chunk_pages` (recursive, page-aware, overlapping) →
`embeddings.embed_documents` (BGE) → `vector_store.add` (Chroma) → persist
`DocumentChunk` rows for citation lookups.

**Query** (`rag/retriever.py`): query rewrite → dense top-k from Chroma →
**BM25 fusion via Reciprocal Rank Fusion** → cross-encoder-style **re-rank** →
context **compression** (dedupe + length cap) → `citations.build_citations`.

## 5. Agent graph

```
START → supervisor ─┬─▶ retrieval ─┐
                    ├─▶ faq ───────┤
                    ├─▶ memory ────┼─▶ response → END
                    └─▶ analytics ─┘
```

Nodes (`agents/nodes.py`) are pure `state → partial-state` functions, so the
**same** nodes power the LangGraph build and the sequential fallback
(`agents/graph.py`). Routing is heuristic by default and can be swapped for an
LLM router without changing the graph shape.

## 6. Database schema (ER)

```
users ─1:N─ chats ─1:N─ messages ─1:N─ feedback
  │                                  
  ├─1:N─ documents ─1:N─ document_chunks   (chunk.vector_id → ChromaDB)
  ├─1:N─ user_sessions
  └─1:N─ analytics_events
```

Indexes: `users.email` (unique), foreign keys on all child tables,
`messages.role`, `documents.status`, `analytics_events.event_type`,
`document_chunks.vector_id`. UUID string PKs keep SQLite↔PostgreSQL portable.

## 7. Deployment topology

Three containers via Compose: `db` (PostgreSQL), `backend` (FastAPI/uvicorn,
persistent volume for Chroma + uploads), `frontend` (Streamlit). The frontend
talks to the backend over the internal Docker network (`BACKEND_URL`).

## 8. Key design decisions

- **OpenAI-compatible LLM client** rather than a hard SDK lock-in — one config
  switch targets OpenAI, Groq, Together, vLLM, or local Ollama.
- **Vectors in Chroma, metadata in SQL** — keeps citations relational and
  per-user filtering cheap while vectors stay in a purpose-built store.
- **RRF hybrid fusion** — robust, parameter-light way to combine dense + sparse
  rankings without tuning score scales.
- **String UUID PKs** — portable and non-enumerable across DB engines.
