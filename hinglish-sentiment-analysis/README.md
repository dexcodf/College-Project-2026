# 🗣️ Hinglish Voice & Text Sentiment Analysis System using Transformers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dexcodf/College-Project-2026/blob/feature/hinglish-sentiment/hinglish-sentiment-analysis/notebooks/Hinglish_Sentiment_Colab.ipynb)

End-to-end Deep Learning NLP system that classifies **Hinglish (Hindi-English
code-mixed)** text and **voice** into **Positive / Negative / Neutral** sentiment.

> 🚀 **Run on a free GPU:** click the *Open in Colab* badge above to clone this
> repo and train on a Colab T4 GPU — see `notebooks/Hinglish_Sentiment_Colab.ipynb`.

- **Text** → Transformer (IndicBERT / mBERT / DistilBERT) → sentiment
- **Voice** → Whisper Speech-to-Text → Transformer → sentiment
- Baseline comparison: **TF-IDF + Logistic Regression**
- Ships with a **FastAPI** backend, a **Streamlit** app, and **Docker** support.

> Target use cases: customer feedback analysis, social-media monitoring,
> product-review analysis, voice-based sentiment.

---

## 📐 Architecture

```
                         ┌─────────────────────────────┐
   Text Input ──────────►│         Tokenizer           │
                         │  (HF AutoTokenizer)         │
   Voice Input           └──────────────┬──────────────┘
       │                                │
       ▼                                ▼
 ┌───────────┐               ┌─────────────────────────┐
 │  Whisper  │── text ──────►│   Transformer Encoder   │
 │   (STT)   │               │ IndicBERT / mBERT / DB  │
 └───────────┘               └──────────────┬──────────┘
                                            │ [CLS] pooled
                                            ▼
                                      ┌───────────┐
                                      │  Dropout  │
                                      └─────┬─────┘
                                            ▼
                                   ┌──────────────────┐
                                   │ Dense (Linear)   │
                                   └────────┬─────────┘
                                            ▼
                                       ┌─────────┐
                                       │ Softmax │
                                       └────┬────┘
                                            ▼
                              Positive / Negative / Neutral
```

### Component / data flow

```
data/            HuggingFace SentiMix ─► dataset_loader ─► preprocessing ─► splits
                                                  │
saved_models/    train.py ─► best_model ◄─────────┘
                     │
reports/         evaluate.py ─► metrics, confusion matrix, curves
                     │
src/inference.py ◄───┘ ──► api/main.py (FastAPI)  ──► streamlit_app/app.py
                          └► src/speech.py (Whisper)
```

---

## 🗂️ Project Structure

```
hinglish-sentiment-analysis/
├── data/                 # raw + processed splits (gitignored)
├── notebooks/
│   └── 01_data_analysis.py   # EDA (script / Jupyter cells)
├── src/
│   ├── config.py             # paths, labels, hyper-params, device select
│   ├── dataset_loader.py     # load + clean + EDA + split
│   ├── preprocessing.py      # Hinglish text cleaning
│   ├── model.py              # transformer + classification head + Dataset
│   ├── train.py              # training loop, curves, early stopping
│   ├── evaluate.py           # report, confusion matrix, macro/weighted F1
│   ├── baseline.py           # TF-IDF + LogisticRegression comparison
│   ├── inference.py          # SentimentPredictor (cached singleton)
│   └── speech.py             # Whisper speech-to-text
├── api/
│   └── main.py               # FastAPI: /predict, /predict-batch, /predict-audio
├── streamlit_app/
│   └── app.py                # Text + Audio UI with confidence viz
├── saved_models/             # trained weights (gitignored)
├── reports/                  # plots + metrics json
├── tests/                    # pytest unit + smoke tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Requires **Python 3.10+**. `ffmpeg` is needed for Whisper audio decoding.

```bash
# 1. Clone & enter
cd hinglish-sentiment-analysis

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install ffmpeg (for voice input)
#   macOS:        brew install ffmpeg
#   Ubuntu:       sudo apt install ffmpeg
#   Windows:      choco install ffmpeg   (or download from ffmpeg.org)
```

> **Apple Silicon / Colab T4**: the code auto-selects the best device
> (CUDA → Apple MPS → CPU) via `src/config.py:get_device()`. No changes needed.

---

## 🏋️ Training

```bash
# 1. (optional) Build splits + EDA visualisations only
python -m src.dataset_loader

# 2. Train the transformer (auto-builds splits if missing, then evaluates)
python -m src.train

# Override hyper-parameters via env vars:
EPOCHS=3 BATCH_SIZE=16 SUBSET_SIZE=20000 MODEL_NAME=bert-base-multilingual-cased \
  python -m src.train

# 3. Train + compare the classical baseline
python -m src.baseline

# 4. (re)evaluate a saved model on the test split
python -m src.evaluate
```

Artifacts produced:

| Path | Contents |
|------|----------|
| `saved_models/best_model/` | weights, tokenizer, `model_meta.json` |
| `reports/training_curves.png` | train/val loss + accuracy curves |
| `reports/confusion_matrix_*.png` | confusion matrices |
| `reports/metrics_*.json` | accuracy, precision, recall, macro/weighted F1 |
| `reports/model_comparison.csv` | baseline vs transformer |

**Target accuracy:** > 85% (achievable with IndicBERT on ~30k samples).

---

## 🔌 Inference & API

### Python
```python
from src.inference import get_predictor
print(get_predictor().predict("Bhai app bahut acha hai"))
# {'sentiment': 'Positive', 'confidence': 0.94, 'probabilities': {...}, 'text': ...}
```

### FastAPI
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Docs: http://localhost:8000/docs
```

**POST `/predict`**
```json
// request
{ "text": "Bhai app bahut acha hai" }
// response
{ "sentiment": "Positive", "confidence": 0.94,
  "probabilities": {"Negative": 0.02, "Neutral": 0.04, "Positive": 0.94},
  "text": "Bhai app bahut acha hai" }
```

**POST `/predict-audio`** — multipart upload (`wav`/`mp3`/`m4a`):
```bash
curl -F "file=@review.m4a" http://localhost:8000/predict-audio
```

---

## 🖥️ Streamlit App

```bash
# In-process model (default)
streamlit run streamlit_app/app.py

# Or route through the FastAPI backend
API_URL=http://localhost:8000 streamlit run streamlit_app/app.py
```

Features: text box with example chips, audio upload (Whisper transcription),
predicted sentiment, and per-class confidence bars.

---

## 🐳 Deployment (Docker)

```bash
# Build & run both services (API :8000 + Streamlit :8501)
docker compose up --build

# API:        http://localhost:8000/docs
# Streamlit:  http://localhost:8501
```

The `saved_models/` directory is mounted into both containers, so train locally
(or copy a trained model in) before starting, and the HuggingFace cache is
shared via a named volume.

---

## 🧪 Tests

```bash
pytest -q
```

Preprocessing tests always run; inference smoke tests auto-skip until a model is
trained.

---

## 📊 Dataset

- **Source:** [`RTT1/SentiMix`](https://huggingface.co/datasets/RTT1/SentiMix) (HuggingFace)
- **~565k** Hinglish records, labels Positive / Negative / Neutral
- The loader is column-name tolerant and merges all splits, then re-splits with
  stratification. For fast iteration it subsamples to `SUBSET_SIZE` (default 30k).
- EDA covers missing values, duplicates, class distribution, and text-length
  distribution (`reports/eda_report.json` + PNGs).

**Challenges handled:** code-mixing, transliteration variation, informal slang,
social-media noise (URLs/mentions/hashtags/emoji/char floods).

---

## 🧩 Model Choices

| Model | HF id | Notes |
|-------|-------|-------|
| IndicBERT (default) | `ai4bharat/indic-bert` | best for Indic + Hinglish |
| Multilingual BERT | `bert-base-multilingual-cased` | strong general baseline |
| DistilBERT (multi) | `distilbert-base-multilingual-cased` | fastest / lightest |

Switch via the `MODEL_NAME` env var (no code change required).

---

## 🛠️ Tech Stack

PyTorch · Transformers · Datasets · scikit-learn · Whisper · FastAPI ·
Streamlit · Docker · Matplotlib/Seaborn · Joblib

---

## 📄 License

For educational / internship demonstration use.
