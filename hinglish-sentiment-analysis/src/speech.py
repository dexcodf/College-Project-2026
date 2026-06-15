"""
Speech-to-Text using OpenAI Whisper.

    Voice Input (wav/mp3/m4a) -> Whisper -> text -> SentimentPredictor

Whisper is loaded lazily and cached so the (sizeable) model is only pulled into
memory when audio is actually submitted.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict

SUPPORTED_FORMATS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm")
# tiny | base | small | medium | large  (base is a good speed/quality trade-off)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")


@lru_cache(maxsize=2)
def _load_whisper(model_size: str = WHISPER_MODEL):
    import whisper  # imported lazily; heavy dependency
    print(f"[speech] Loading Whisper '{model_size}' model ...")
    return whisper.load_model(model_size)


def transcribe(audio_path: str, model_size: str = WHISPER_MODEL,
               language: str | None = None) -> Dict:
    """
    Transcribe an audio file to text.

    Parameters
    ----------
    audio_path : str
        Path to a wav/mp3/m4a/... file.
    language : str | None
        Optional language hint (e.g. "hi" or "en"). None = auto-detect, which
        works well for code-mixed Hinglish speech.

    Returns
    -------
    dict with keys: text, language, segments(count).
    """
    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported audio format '{ext}'. Supported: {SUPPORTED_FORMATS}"
        )

    model = _load_whisper(model_size)
    result = model.transcribe(audio_path, language=language, fp16=False)
    return {
        "text": result.get("text", "").strip(),
        "language": result.get("language"),
        "segments": len(result.get("segments", [])),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.speech <audio_file>")
        raise SystemExit(1)
    out = transcribe(sys.argv[1])
    print(f"[{out['language']}] {out['text']}")
