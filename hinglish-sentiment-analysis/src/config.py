"""
Central configuration for the Hinglish Sentiment Analysis system.

All paths, hyper-parameters and label mappings live here so that the
data / model / api / streamlit layers stay in sync.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "saved_models"
REPORTS_DIR = PROJECT_ROOT / "reports"

for _d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
# Canonical order used everywhere (id <-> label).
LABELS = ["Negative", "Neutral", "Positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
NUM_LABELS = len(LABELS)

# Common raw-label spellings found across SentiMix-style datasets -> canonical.
RAW_LABEL_ALIASES = {
    "positive": "Positive",
    "pos": "Positive",
    "1": "Positive",
    "2": "Positive",  # some splits use 0/1/2 = neg/neu/pos
    "negative": "Negative",
    "neg": "Negative",
    "0": "Negative",
    "neutral": "Neutral",
    "neu": "Neutral",
    "neut": "Neutral",
}


# --------------------------------------------------------------------------- #
# Device selection (CUDA / Apple MPS / CPU)
# --------------------------------------------------------------------------- #
def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


# --------------------------------------------------------------------------- #
# Model / training hyper-parameters
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    # HuggingFace model id. Alternatives:
    #   "distilbert-base-multilingual-cased" -> faster / lighter (default, ungated)
    #   "bert-base-multilingual-cased"       -> mBERT (stronger, heavier)
    #   "ai4bharat/indic-bert"               -> IndicBERT (best for Hinglish, but
    #                                            GATED: needs `huggingface-cli login`)
    model_name: str = os.getenv("MODEL_NAME", "distilbert-base-multilingual-cased")

    # Dataset
    dataset_name: str = os.getenv("DATASET_NAME", "RTT1/SentiMix")
    subset_size: int = int(os.getenv("SUBSET_SIZE", "30000"))  # 20k-50k recommended
    max_length: int = int(os.getenv("MAX_LENGTH", "128"))
    test_size: float = 0.10
    val_size: float = 0.10
    random_seed: int = 42

    # Optimisation
    epochs: int = int(os.getenv("EPOCHS", "4"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "32"))
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    dropout: float = 0.3
    max_grad_norm: float = 1.0

    # Bookkeeping
    output_dir: Path = field(default=MODEL_DIR)
    early_stopping_patience: int = 2

    @property
    def best_model_path(self) -> Path:
        return self.output_dir / "best_model"


CONFIG = TrainConfig()
