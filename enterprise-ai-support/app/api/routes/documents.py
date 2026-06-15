"""Document upload, listing, and deletion."""
from __future__ import annotations

from fastapi import APIRouter, File, UploadFile, status
from sqlalchemy import select

from app.dependencies import CurrentUser, DbSession
from app.exceptions import NotFoundError, PermissionError_, ValidationError_
from app.models.document import Document, DocumentStatus
from app.rag.loaders import SUPPORTED_EXTENSIONS
from app.rag.pipeline import delete_document_vectors, ingest_document, save_upload
from app.schemas.document import DocumentOut, IngestResult
from pathlib import Path

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB


@router.post("", response_model=IngestResult, status_code=status.HTTP_201_CREATED)
async def upload_document(
    user: CurrentUser, db: DbSession, file: UploadFile = File(...)
) -> IngestResult:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValidationError_(
            f"Unsupported file type '{ext}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    data = await file.read()
    if not data:
        raise ValidationError_("Uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValidationError_("File exceeds the 25 MB limit")

    path = save_upload(file.filename or "upload", data, user.id)
    document = Document(
        owner_id=user.id,
        filename=Path(file.filename or path.name).name,
        content_type=file.content_type or "",
        size_bytes=len(data),
        status=DocumentStatus.PENDING,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    document = ingest_document(db, document, path)
    return IngestResult(
        document=DocumentOut.model_validate(document),
        chunks_created=document.chunk_count,
    )


@router.get("", response_model=list[DocumentOut])
def list_documents(user: CurrentUser, db: DbSession) -> list[DocumentOut]:
    rows = db.scalars(
        select(Document)
        .where(Document.owner_id == user.id)
        .order_by(Document.created_at.desc())
    ).all()
    return [DocumentOut.model_validate(d) for d in rows]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, user: CurrentUser, db: DbSession) -> None:
    document = db.get(Document, document_id)
    if document is None:
        raise NotFoundError("Document not found")
    if document.owner_id != user.id:
        raise PermissionError_("You do not own this document")
    delete_document_vectors(document.id)
    db.delete(document)
    db.commit()
