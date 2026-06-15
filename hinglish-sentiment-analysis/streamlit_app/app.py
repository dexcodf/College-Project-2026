"""
Streamlit frontend for Hinglish Voice & Text Sentiment Analysis.

Two modes:
  * Text input  -> direct sentiment prediction
  * Audio upload -> Whisper transcription -> sentiment prediction

By default it talks to the model in-process (via src.inference). Set
API_URL to route through the FastAPI backend instead.

Run:
    streamlit run streamlit_app/app.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Make `src` importable when run from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

API_URL = os.getenv("API_URL")  # e.g. "http://localhost:8000"; None = in-process

SENTIMENT_COLORS = {"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"}
SENTIMENT_EMOJI = {"Positive": "😊", "Neutral": "😐", "Negative": "😠"}

st.set_page_config(page_title="Hinglish Sentiment Analysis", page_icon="🗣️",
                   layout="centered")


# --------------------------------------------------------------------------- #
# Prediction helpers (in-process or via API)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading sentiment model ...")
def _load_predictor():
    from src.inference import get_predictor
    return get_predictor()


def predict_text(text: str) -> dict:
    if API_URL:
        import requests
        r = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=60)
        r.raise_for_status()
        return r.json()
    return _load_predictor().predict(text)


def predict_audio(file_bytes: bytes, suffix: str) -> dict:
    if API_URL:
        import requests
        files = {"file": (f"audio{suffix}", file_bytes)}
        r = requests.post(f"{API_URL}/predict-audio", files=files, timeout=300)
        r.raise_for_status()
        return r.json()
    # In-process: write temp file, transcribe, predict.
    from src.speech import transcribe
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        stt = transcribe(tmp_path)
        pred = _load_predictor().predict(stt["text"])
        return {"transcription": stt["text"],
                "detected_language": stt["language"], **pred}
    finally:
        os.remove(tmp_path)


# --------------------------------------------------------------------------- #
# UI rendering
# --------------------------------------------------------------------------- #
def render_result(result: dict):
    sentiment = result["sentiment"]
    color = SENTIMENT_COLORS.get(sentiment, "#333")
    emoji = SENTIMENT_EMOJI.get(sentiment, "")

    st.markdown(
        f"<h2 style='color:{color};'>{emoji} {sentiment} "
        f"<span style='font-size:0.6em;color:#777;'>"
        f"({result['confidence']*100:.1f}% confidence)</span></h2>",
        unsafe_allow_html=True,
    )

    probs = result.get("probabilities", {})
    if probs:
        st.subheader("Confidence by class")
        for label in ["Negative", "Neutral", "Positive"]:
            p = probs.get(label, 0.0)
            st.markdown(f"**{label}** — {p*100:.1f}%")
            st.progress(min(max(p, 0.0), 1.0))


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
st.title("🗣️ Hinglish Sentiment Analysis")
st.caption("Sentiment classification for Hindi-English code-mixed text & voice "
           "using Transformers + Whisper.")

if API_URL:
    st.info(f"Routing predictions through API: {API_URL}")

tab_text, tab_audio = st.tabs(["📝 Text Input", "🎤 Voice Input"])

with tab_text:
    text = st.text_area(
        "Enter Hinglish text",
        placeholder="e.g. Bhai app bahut acha hai",
        height=120,
    )
    examples = ["Bhai app bahut acha hai",
                "Yeh movie bilkul bakwaas thi",
                "Theek hai, kuch khaas nahi"]
    cols = st.columns(len(examples))
    for c, ex in zip(cols, examples):
        if c.button(ex, use_container_width=True):
            text = ex

    if st.button("Analyze Sentiment", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing ..."):
                try:
                    render_result(predict_text(text))
                except Exception as e:  # noqa: BLE001
                    st.error(f"Prediction failed: {e}")

with tab_audio:
    st.write("Upload an audio clip (WAV / MP3 / M4A). It will be transcribed with "
             "Whisper, then analysed.")
    audio_file = st.file_uploader("Audio file", type=["wav", "mp3", "m4a", "flac", "ogg"])
    if audio_file is not None:
        st.audio(audio_file)
        if st.button("Transcribe & Analyze", type="primary", use_container_width=True):
            suffix = os.path.splitext(audio_file.name)[1].lower()
            with st.spinner("Transcribing & analyzing (first run downloads Whisper) ..."):
                try:
                    result = predict_audio(audio_file.getvalue(), suffix)
                    st.success("Transcription:")
                    st.write(f"> {result.get('transcription', '')}")
                    if result.get("detected_language"):
                        st.caption(f"Detected language: {result['detected_language']}")
                    render_result(result)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Prediction failed: {e}")

st.divider()
st.caption("Model: IndicBERT / mBERT (configurable) • STT: OpenAI Whisper")
