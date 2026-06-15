"""LLM client wrapper around any OpenAI-compatible chat endpoint.

Works with OpenAI, Groq, Together, vLLM, and local Ollama by pointing
``LLM_BASE_URL`` at the right host. When no API key is configured the client
degrades gracefully to an *extractive* answer built from retrieved context,
so the application remains fully runnable in demos and CI without a key.
"""
from __future__ import annotations

from collections.abc import Iterator

from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.logging_config import get_logger

logger = get_logger("llm")

ChatMessages = list[dict[str, str]]


class LLMClient:
    """Thin, retrying wrapper over an OpenAI-compatible chat completions API."""

    def __init__(self) -> None:
        self._client = None
        if settings.llm_enabled:
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=settings.llm_api_key, base_url=settings.llm_base_url
                )
            except Exception as exc:  # pragma: no cover - import/config issues
                logger.warning("llm_init_failed", error=str(exc))
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    )
    def complete(self, messages: ChatMessages, *, temperature: float | None = None) -> str:
        """Return a full completion string."""
        if not self.available:
            return self._fallback(messages)
        resp = self._client.chat.completions.create(  # type: ignore[union-attr]
            model=settings.llm_model,
            messages=messages,
            temperature=settings.llm_temperature if temperature is None else temperature,
        )
        return resp.choices[0].message.content or ""

    def stream(self, messages: ChatMessages, *, temperature: float | None = None) -> Iterator[str]:
        """Yield completion tokens as they arrive."""
        if not self.available:
            yield self._fallback(messages)
            return
        stream = self._client.chat.completions.create(  # type: ignore[union-attr]
            model=settings.llm_model,
            messages=messages,
            temperature=settings.llm_temperature if temperature is None else temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    @staticmethod
    def _fallback(messages: ChatMessages) -> str:
        """Deterministic, no-API answer assembled from the provided context.

        The agent layer injects retrieved context into a system message; here
        we surface it so the product still answers (with citations) offline.
        """
        context = next(
            (m["content"] for m in messages if m["role"] == "system" and "CONTEXT" in m["content"]),
            "",
        )
        question = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        if context:
            snippet = context.split("CONTEXT:", 1)[-1].strip()[:900]
            return (
                "Based on the knowledge base, here is the most relevant "
                f"information for your question:\n\n{snippet}\n\n"
                "_(LLM API not configured — showing retrieved context. Set "
                "`LLM_API_KEY` for fully synthesised answers.)_"
            )
        return (
            f"I received your question: “{question}”. No knowledge-base context "
            "was found and no LLM API key is configured, so I can't synthesise a "
            "full answer yet. Upload documents or configure `LLM_API_KEY`."
        )


# Process-wide singleton.
llm_client = LLMClient()
