"""
Inference layer: a singleton-style predictor reused by the API and Streamlit app.

    from src.inference import get_predictor
    get_predictor().predict("Bhai app bahut acha hai")
    # -> {"sentiment": "Positive", "confidence": 0.94, "probabilities": {...}}
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from .config import CONFIG, DEVICE, ID2LABEL
from .model import HinglishSentimentClassifier
from .preprocessing import clean_text


class SentimentPredictor:
    """Loads the trained transformer once and serves predictions."""

    def __init__(self, model_dir: Path = None, device: torch.device = None):
        self.model_dir = Path(model_dir or CONFIG.best_model_path)
        self.device = device or DEVICE
        if not (self.model_dir / "model_meta.json").exists():
            raise FileNotFoundError(
                f"No trained model at {self.model_dir}. Run `python -m src.train` first."
            )
        self.model, self.tokenizer, self.meta = HinglishSentimentClassifier.load(
            self.model_dir, map_location=self.device
        )
        self.model.to(self.device).eval()
        self.id2label = {int(k): v for k, v in self.meta.get("id2label", ID2LABEL).items()}
        self.max_length = self.meta.get("max_length", CONFIG.max_length)

    @torch.no_grad()
    def predict(self, text: str) -> Dict:
        """Predict sentiment for a single string."""
        return self.predict_batch([text])[0]

    @torch.no_grad()
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict sentiment for a list of strings."""
        cleaned = [clean_text(t) for t in texts]
        enc = self.tokenizer(
            cleaned, truncation=True, padding=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(self.device)
        logits = self.model(**enc)["logits"]
        probs = F.softmax(logits, dim=-1).cpu()

        results = []
        for i, p in enumerate(probs):
            conf, idx = torch.max(p, dim=-1)
            results.append({
                "text": texts[i],
                "sentiment": self.id2label[int(idx)],
                "confidence": round(float(conf), 4),
                "probabilities": {
                    self.id2label[j]: round(float(p[j]), 4) for j in range(len(p))
                },
            })
        return results


@lru_cache(maxsize=1)
def get_predictor(model_dir: str = None) -> SentimentPredictor:
    """Cached accessor so the model is loaded only once per process."""
    return SentimentPredictor(model_dir=model_dir)


if __name__ == "__main__":
    samples = [
        "Bhai app bahut acha hai",
        "Yeh movie bilkul bakwaas thi, paise barbaad",
        "Theek hai, kuch khaas nahi",
    ]
    predictor = get_predictor()
    for r in predictor.predict_batch(samples):
        print(f"{r['sentiment']:>8}  ({r['confidence']:.2f})  | {r['text']}")
