"""Central application configuration.

Settings are loaded from environment variables (and an optional `.env` file)
using pydantic-settings. Import the singleton ``settings`` everywhere rather
than reading ``os.environ`` directly — this keeps configuration validated and
typed in one place (SOLID: single source of truth).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = directory that contains this `app/` package's parent.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"


class Settings(BaseSettings):
    """Typed application settings."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- App ----
    app_name: str = "Enterprise AI Support"
    environment: str = "development"
    debug: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ---- Security ----
    secret_key: str = "change-me-in-production"
    access_token_expire_minutes: int = 60
    algorithm: str = "HS256"

    # ---- Database ----
    database_url: str = f"sqlite:///{DATA_DIR / 'app.db'}"

    # ---- LLM ----
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2

    # ---- Embeddings ----
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_device: str = "cpu"

    # ---- Vector store ----
    chroma_persist_dir: str = str(DATA_DIR / "chroma")
    chroma_collection: str = "support_docs"

    # ---- RAG ----
    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_top_k: int = 8
    rerank_top_n: int = 4

    # ---- Voice ----
    whisper_model: str = "base"

    # ---- Frontend ----
    backend_url: str = "http://localhost:8000"

    # ---- Derived helpers ----
    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def llm_enabled(self) -> bool:
        """Whether a usable LLM endpoint is configured."""
        return bool(self.llm_api_key)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (created once per process)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return Settings()


settings = get_settings()
