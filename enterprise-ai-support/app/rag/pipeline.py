"""Ingestion pipeline: persist a file, extract, chunk, embed, index, record.

This is the write-path counterpart to ``retriever.retrieve``. It is
transactional with respect to the Document row's status so a failed ingest is
visible in the UI and re-runnable.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import UPLOAD_DIR
from app.logging_config import get_logger
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.rag.chunking import chunk_pages
from app.rag.loaders import load_document
from app.rag.vector_store import vector_store

logger = get_logger("rag.pipeline")


def save_upload(filename: str, data: bytes, owner_id: str) -> Path:
    """Persist raw upload bytes to a per-owner directory; return the path."""
    safe_name = Path(filename).name
    owner_dir = UPLOAD_DIR / owner_id
    owner_dir.mkdir(parents=True, exist_ok=True)
    dest = owner_dir / f"{uuid.uuid4().hex}_{safe_name}"
    dest.write_bytes(data)
    return dest


def ingest_document(db: Session, document: Document, file_path: Path) -> Document:
    """Process a stored file into vectors + chunk metadata.

    Updates ``document.status`` through PROCESSING → READY/FAILED.
    """
    document.status = DocumentStatus.PROCESSING
    db.commit()

    try:
        pages = load_document(file_path)
        chunks = chunk_pages(pages)
        if not chunks:
            raise ValueError("Document produced no chunks")

        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict] = []
        chunk_rows: list[DocumentChunk] = []

        for chunk in chunks:
            vector_id = uuid.uuid4().hex
            ids.append(vector_id)
            texts.append(chunk.text)
            metadatas.append(
                {
                    "document_id": document.id,
                    "owner_id": document.owner_id,
                    "filename": document.filename,
                    "page": chunk.page if chunk.page is not None else 0,
                    "chunk_index": chunk.index,
                }
            )
            chunk_rows.append(
                DocumentChunk(
                    document_id=document.id,
                    chunk_index=chunk.index,
                    vector_id=vector_id,
                    page=chunk.page,
                    text=chunk.text,
                )
            )

        vector_store.add(ids=ids, texts=texts, metadatas=metadatas)
        db.add_all(chunk_rows)

        document.chunk_count = len(chunk_rows)
        document.status = DocumentStatus.READY
        document.error = None
        db.commit()
        db.refresh(document)
        logger.info("ingest_ok", document_id=document.id, chunks=len(chunk_rows))
        return document

    except Exception as exc:
        db.rollback()
        document.status = DocumentStatus.FAILED
        document.error = str(exc)[:1000]
        db.commit()
        logger.error("ingest_failed", document_id=document.id, error=str(exc))
        return document


def delete_document_vectors(document_id: str) -> None:
    """Remove a document's vectors from the store (chunks cascade in the DB)."""
    vector_store.delete_document(document_id)
