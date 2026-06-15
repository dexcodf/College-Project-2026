"""
Classical baseline: TF-IDF + Logistic Regression.

Provides a reference point to compare against the transformer. Run after the
processed splits exist (or it will build them):

    python -m src.baseline
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .config import CONFIG, LABEL2ID, MODEL_DIR, REPORTS_DIR
from .evaluate import evaluate_predictions


def _load_splits():
    from .config import PROCESSED_DIR
    files = [PROCESSED_DIR / f"{s}.csv" for s in ("train", "val", "test")]
    if not all(f.exists() for f in files):
        from .dataset_loader import build_datasets
        return build_datasets(save=True)
    return tuple(pd.read_csv(f) for f in files)


def train_baseline() -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    train_df, _val_df, test_df = _load_splits()

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced")),
    ])

    print("[baseline] Fitting TF-IDF + LogisticRegression ...")
    pipe.fit(train_df["text"], train_df["label"].map(LABEL2ID))

    y_true = test_df["label"].map(LABEL2ID).to_numpy()
    y_pred = pipe.predict(test_df["text"])
    metrics = evaluate_predictions(y_true, y_pred, tag="baseline")

    model_path = MODEL_DIR / "baseline_tfidf_logreg.joblib"
    joblib.dump(pipe, model_path)
    print(f"[baseline] Saved baseline to {model_path}")
    return metrics


def compare():
    """Print a side-by-side comparison of baseline vs transformer (if available)."""
    rows = []
    for tag in ("baseline", "transformer"):
        p = REPORTS_DIR / f"metrics_{tag}.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                m = json.load(f)
            rows.append({"model": tag, "accuracy": m["accuracy"],
                         "f1_macro": m["f1_macro"], "f1_weighted": m["f1_weighted"]})
    if rows:
        comp = pd.DataFrame(rows).set_index("model")
        print("\n===== Model Comparison =====")
        print(comp.round(4).to_string())
        comp.to_csv(REPORTS_DIR / "model_comparison.csv")
    return rows


if __name__ == "__main__":
    train_baseline()
    compare()
