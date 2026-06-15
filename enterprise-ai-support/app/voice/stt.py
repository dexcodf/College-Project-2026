"""Speech-to-text via OpenAI Whisper (local model).

The model is loaded lazily and cached, since loading is expensive. If Whisper
isn't installed the service raises a clear, typed error the API turns into a
503 rather than crashing the process.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from app.config import settings
from app.exceptions import ServiceUnavailableError
from app.logging_config import get_logger

logger = get_logger("voice.stt")

_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            import whisper
        except ImportError as exc:  # pragma: no cover
            raise ServiceUnavailableError(
                "Whisper is not installed on the server"
            ) from exc
        logger.info("whisper_loading", model=settings.whisper_model)
        _model = whisper.load_model(settings.whisper_model)
    return _model


def transcribe(audio_bytes: bytes, *, suffix: str = ".wav") -> str:
    """Transcribe raw audio bytes to text."""
    model = _get_model()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)
    try:
        result = model.transcribe(str(tmp_path))
        return (result.get("text") or "").strip()
    finally:
        tmp_path.unlink(missing_ok=True)
