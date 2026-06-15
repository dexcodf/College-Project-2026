"""Unit tests for the chunking module (pure, no external deps)."""
from __future__ import annotations

from app.rag.chunking import chunk_pages
from app.rag.loaders import LoadedPage


def test_chunk_respects_max_size():
    text = " ".join(f"word{i}" for i in range(500))
    pages = [LoadedPage(text=text, page=1)]
    chunks = chunk_pages(pages, chunk_size=100, overlap=0)
    assert len(chunks) > 1
    assert all(len(c.text) <= 160 for c in chunks)  # overlap-free upper bound


def test_chunk_preserves_page_number():
    pages = [LoadedPage(text="a" * 50, page=7)]
    chunks = chunk_pages(pages, chunk_size=200, overlap=0)
    assert chunks[0].page == 7


def test_chunk_indices_are_sequential():
    pages = [LoadedPage(text="alpha beta gamma. " * 60, page=None)]
    chunks = chunk_pages(pages, chunk_size=80, overlap=10)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_empty_text_yields_no_chunks():
    assert chunk_pages([LoadedPage(text="   ", page=1)]) == []
