"""Semantic-ish recursive chunking.

Splits text on progressively finer separators (paragraphs → sentences →
words) to keep chunks near a target size without cutting mid-sentence, then
applies a sliding overlap so context isn't lost at chunk boundaries.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import settings
from app.rag.loaders import LoadedPage

_SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    text: str
    index: int
    page: int | None


def _split_recursive(text: str, max_size: int) -> list[str]:
    """Recursively split ``text`` into pieces no larger than ``max_size``."""
    text = text.strip()
    if len(text) <= max_size:
        return [text] if text else []

    for sep in _SEPARATORS:
        if sep in text:
            parts = text.split(sep)
            chunks: list[str] = []
            buffer = ""
            for part in parts:
                candidate = f"{buffer}{sep}{part}" if buffer else part
                if len(candidate) <= max_size:
                    buffer = candidate
                else:
                    if buffer:
                        chunks.append(buffer)
                    # A single part may still exceed max_size → recurse.
                    if len(part) > max_size:
                        chunks.extend(_split_recursive(part, max_size))
                        buffer = ""
                    else:
                        buffer = part
            if buffer:
                chunks.append(buffer)
            return [c.strip() for c in chunks if c.strip()]

    # No separator matched — hard split.
    return [text[i : i + max_size] for i in range(0, len(text), max_size)]


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        tail = prev[-overlap:]
        out.append(f"{tail} {cur}".strip())
    return out


def chunk_pages(
    pages: list[LoadedPage],
    *,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Chunk a list of loaded pages into overlapping, page-aware chunks."""
    size = chunk_size or settings.chunk_size
    ov = overlap if overlap is not None else settings.chunk_overlap

    result: list[Chunk] = []
    idx = 0
    for page in pages:
        normalised = re.sub(r"[ \t]+", " ", page.text)
        raw_chunks = _split_recursive(normalised, size)
        for piece in _apply_overlap(raw_chunks, ov):
            result.append(Chunk(text=piece, index=idx, page=page.page))
            idx += 1
    return result
