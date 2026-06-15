# Deployment Guide

## Environments

| Setting | Dev | Production |
|---------|-----|-----------|
| `ENVIRONMENT` | `development` | `production` |
| `DEBUG` | `true` | `false` |
| `DATABASE_URL` | SQLite (default) | PostgreSQL |
| Logs | colored console | JSON |
| CORS | `*` | `BACKEND_URL` only |

## 1. Docker Compose (recommended)

```bash
export SECRET_KEY=$(python -c "import secrets;print(secrets.token_urlsafe(48))")
export LLM_API_KEY=sk-...            # optional but recommended
docker compose up --build -d
```

Services: PostgreSQL (`db`), FastAPI (`backend`, :8000), Streamlit
(`frontend`, :8501). A named volume `appdata` persists ChromaDB + uploads;
`pgdata` persists the database.

Tear down (and wipe volumes): `docker compose down -v`.

## 2. Manual / VM

```bash
pip install -r requirements.txt
# Backend (gunicorn-style workers via uvicorn)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
# Frontend
BACKEND_URL=https://api.example.com streamlit run frontend/streamlit_app.py
```

## 3. PaaS (Render / Fly / Cloud Run)

The backend image honours an injected `$API_PORT`/`$PORT`. Provide env vars:
`SECRET_KEY`, `DATABASE_URL` (managed Postgres), `LLM_API_KEY`,
`CHROMA_PERSIST_DIR` (mount a persistent disk).

## Production checklist

- [ ] Strong random `SECRET_KEY` (48+ bytes); never commit `.env`.
- [ ] Change/disable the seeded `admin@example.com` account.
- [ ] Managed PostgreSQL with backups; run migrations (Alembic) on deploy.
- [ ] Persistent disk mounted at `CHROMA_PERSIST_DIR`.
- [ ] `ENVIRONMENT=production`, `DEBUG=false` (enables JSON logs, locks CORS).
- [ ] Reverse proxy (TLS termination) in front of backend + frontend.
- [ ] Resource sizing: embedding model + Whisper benefit from more RAM/CPU
      (or set `EMBEDDING_DEVICE=cuda` on GPU hosts).
- [ ] Health probe: `GET /api/health`.

## Database migrations

Alembic is included. To initialise:

```bash
alembic init migrations          # one-time
alembic revision --autogenerate -m "init"
alembic upgrade head
```

For the SQLite dev default, `init_db()` auto-creates tables at startup.

## Scaling notes

- Backend is stateless (except local Chroma) → scale horizontally behind a
  load balancer once Chroma is moved to a shared server / Chroma Cloud.
- Move embedding + Whisper inference to a dedicated worker/GPU service for
  high throughput; the interfaces (`EmbeddingProvider`, `voice/*`) are the
  seams to swap.
