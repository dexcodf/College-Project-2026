"""Citation engine — turns retrieved chunks into structured source citations."""
from __future__ import annotations

from app.rag.vector_store import RetrievedChunk
from app.schemas.chat import Citation


def _snippet(text: str, *, max_len: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


def build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    """Map retrieved chunks to deduplicated, ordered Citation objects."""
    citations: list[Citation] = []
    seen: set[tuple[str, int | None]] = set()
    for chunk in chunks:
        meta = chunk.metadata
        key = (meta.get("document_id", ""), meta.get("page"))
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                document_id=meta.get("document_id", ""),
                filename=meta.get("filename", "unknown"),
                page=meta.get("page"),
                score=round(float(chunk.score), 4),
                snippet=_snippet(chunk.text),
            )
        )
    return citations


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered context block for the LLM prompt."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        src = chunk.metadata.get("filename", "unknown")
        page = chunk.metadata.get("page")
        loc = f"{src}, p.{page}" if page else src
        lines.append(f"[{i}] ({loc})\n{chunk.text}")
    return "\n\n".join(lines)
