"""Document loaders with OCR support.

Each loader returns a list of ``LoadedPage`` so downstream chunking can keep
page provenance for citations. Supported: PDF, DOCX, TXT/MD, and images
(PNG/JPG) via Tesseract OCR. Heavy parsers are imported lazily so the module
imports cleanly even if an optional dependency is missing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.exceptions import ValidationError_
from app.logging_config import get_logger

logger = get_logger("rag.loaders")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg"}


@dataclass
class LoadedPage:
    """A unit of extracted text with its 1-based page number (or None)."""

    text: str
    page: int | None


def load_document(path: str | Path) -> list[LoadedPage]:
    """Dispatch to the right loader based on file extension."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValidationError_(f"Unsupported file type: {ext}")

    if ext == ".pdf":
        pages = _load_pdf(path)
    elif ext == ".docx":
        pages = _load_docx(path)
    elif ext in {".png", ".jpg", ".jpeg"}:
        pages = _load_image(path)
    else:  # .txt / .md
        pages = _load_text(path)

    pages = [p for p in pages if p.text.strip()]
    if not pages:
        raise ValidationError_("No extractable text found in document")
    logger.info("document_loaded", file=path.name, pages=len(pages))
    return pages


def _load_text(path: Path) -> list[LoadedPage]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [LoadedPage(text=text, page=None)]


def _load_pdf(path: Path) -> list[LoadedPage]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [
        LoadedPage(text=page.extract_text() or "", page=i + 1)
        for i, page in enumerate(reader.pages)
    ]


def _load_docx(path: Path) -> list[LoadedPage]:
    import docx

    document = docx.Document(str(path))
    text = "\n".join(p.text for p in document.paragraphs)
    return [LoadedPage(text=text, page=None)]


def _load_image(path: Path) -> list[LoadedPage]:
    """Run OCR on an image. Requires the system `tesseract` binary."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ValidationError_("OCR dependencies not installed") from exc

    try:
        text = pytesseract.image_to_string(Image.open(path))
    except Exception as exc:  # pragma: no cover - tesseract not on PATH
        raise ValidationError_(
            "OCR failed — is the Tesseract binary installed?"
        ) from exc
    return [LoadedPage(text=text, page=1)]
