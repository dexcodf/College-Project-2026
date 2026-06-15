"""Voice endpoints: speech-to-text and text-to-speech."""
from __future__ import annotations

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from pathlib import Path

from app.dependencies import CurrentUser
from app.voice.stt import transcribe
from app.voice.tts import synthesize

router = APIRouter(prefix="/voice", tags=["voice"])


class TranscriptOut(BaseModel):
    text: str


@router.post("/transcribe", response_model=TranscriptOut)
async def transcribe_audio(
    _: CurrentUser, file: UploadFile = File(...)
) -> TranscriptOut:
    data = await file.read()
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    text = transcribe(data, suffix=suffix)
    return TranscriptOut(text=text)


class SpeakRequest(BaseModel):
    text: str
    lang: str = "en"


@router.post("/speak")
def speak(payload: SpeakRequest, _: CurrentUser) -> Response:
    audio = synthesize(payload.text, lang=payload.lang)
    return Response(content=audio, media_type="audio/mpeg")
