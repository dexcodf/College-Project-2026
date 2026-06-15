"""Text-to-speech via gTTS, returning MP3 bytes."""
from __future__ import annotations

import io

from app.exceptions import ServiceUnavailableError
from app.logging_config import get_logger

logger = get_logger("voice.tts")


def synthesize(text: str, *, lang: str = "en") -> bytes:
    """Convert text to spoken MP3 audio bytes."""
    if not text.strip():
        raise ServiceUnavailableError("No text provided for synthesis")
    try:
        from gtts import gTTS
    except ImportError as exc:  # pragma: no cover
        raise ServiceUnavailableError("gTTS is not installed on the server") from exc

    buffer = io.BytesIO()
    gTTS(text=text[:5000], lang=lang).write_to_fp(buffer)
    buffer.seek(0)
    return buffer.read()
