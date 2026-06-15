"""
FastAPI backend for Hinglish sentiment analysis.

Endpoints
---------
GET  /                 -> service metadata
GET  /health           -> health + model-loaded status
POST /predict          -> sentiment for a JSON text payload
POST /predict-batch    -> sentiment for a list of texts
POST /predict-audio    -> transcribe (Whisper) + sentiment for an uploaded file

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import os
import shutil
import tempfile
from typing import Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Hinglish Sentiment Analysis API",
    description="Sentiment classification for Hindi-English code-mixed text & voice.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, example="Bhai app bahut acha hai")


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1)


class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]
    text: str


# --------------------------------------------------------------------------- #
# Lazy model accessors (so the app boots even before a model is trained)
# --------------------------------------------------------------------------- #
def _predictor():
    from src.inference import get_predictor
    try:
        return get_predictor()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def root():
    return {
        "service": "Hinglish Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": ["/predict", "/predict-batch", "/predict-audio", "/health"],
    }


@app.get("/health")
def health():
    from src.config import CONFIG
    model_ready = (CONFIG.best_model_path / "model_meta.json").exists()
    return {"status": "ok", "model_loaded": model_ready,
            "model_dir": str(CONFIG.best_model_path)}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: TextRequest):
    result = _predictor().predict(req.text)
    return result


@app.post("/predict-batch", response_model=List[PredictionResponse])
def predict_batch(req: BatchRequest):
    return _predictor().predict_batch(req.texts)


@app.post("/predict-audio")
async def predict_audio(file: UploadFile = File(...)):
    """Transcribe an uploaded audio file with Whisper, then classify sentiment."""
    from src.speech import SUPPORTED_FORMATS, transcribe

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported: {SUPPORTED_FORMATS}",
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        stt = transcribe(tmp_path)
        if not stt["text"]:
            raise HTTPException(status_code=422, detail="No speech detected in audio.")

        prediction = _predictor().predict(stt["text"])
        return {
            "transcription": stt["text"],
            "detected_language": stt["language"],
            **prediction,
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
