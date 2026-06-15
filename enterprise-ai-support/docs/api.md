# API Reference

Base URL: `http://localhost:8000`. Interactive docs: `/docs` (Swagger),
`/redoc`. All `/api/...` endpoints except auth + health require a
`Authorization: Bearer <token>` header.

## Auth

| Method | Path | Body | Notes |
|--------|------|------|-------|
| POST | `/api/auth/signup` | `{email, password, full_name}` | → `{access_token, user}` (201) |
| POST | `/api/auth/login` | `{email, password}` | → `{access_token, user}` |
| GET | `/api/auth/me` | — | current user |

## Documents

| Method | Path | Body | Notes |
|--------|------|------|-------|
| POST | `/api/documents` | multipart `file` | ingest (PDF/DOCX/TXT/MD/PNG/JPG), ≤25 MB |
| GET | `/api/documents` | — | list own documents |
| DELETE | `/api/documents/{id}` | — | delete doc + vectors (204) |

## Chat

| Method | Path | Body | Notes |
|--------|------|------|-------|
| POST | `/api/chat/ask` | `{message, chat_id?, top_k?}` | → `{chat_id, message}` |
| GET | `/api/chat/chats` | — | list conversations |
| GET | `/api/chat/chats/{id}` | — | conversation with messages |
| DELETE | `/api/chat/chats/{id}` | — | delete conversation (204) |
| POST | `/api/chat/messages/{id}/feedback` | `{rating: up\|down, comment?}` | rate an answer (201) |

## Analytics (admin only)

| Method | Path | Query | Notes |
|--------|------|-------|-------|
| GET | `/api/analytics/overview` | `days` | KPIs + time series + routing |

## Memory & Settings

| Method | Path | Body |
|--------|------|------|
| GET | `/api/memory/preferences` | — |
| PUT | `/api/memory/preferences` | `{theme, voice_enabled, default_top_k}` |
| GET | `/api/settings/public` | — |
| PUT | `/api/settings/profile` | `{full_name}` |

## Voice

| Method | Path | Body | Notes |
|--------|------|------|-------|
| POST | `/api/voice/transcribe` | multipart `file` | → `{text}` (Whisper) |
| POST | `/api/voice/speak` | `{text, lang}` | → `audio/mpeg` (gTTS) |

## Errors

All errors return `{"error": {"code": "...", "message": "..."}}` with an
appropriate HTTP status (400/401/403/404/409/422/503/500).

### Example

```bash
TOKEN=$(curl -s localhost:8000/api/auth/login \
  -H 'content-type: application/json' \
  -d '{"email":"admin@example.com","password":"admin12345"}' | jq -r .access_token)

curl -s localhost:8000/api/chat/ask -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"message":"What is your refund policy?"}' | jq
```
