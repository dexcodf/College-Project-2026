"""
Evaluation utilities: classification report, confusion matrix, macro/weighted F1.

    python -m src.evaluate                 # evaluate saved best model on test split
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import CONFIG, DEVICE, LABEL2ID, LABELS, REPORTS_DIR
from .model import HinglishSentimentClassifier, SentimentDataset


def _predict_logits(model, loader):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"]
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch)["logits"].cpu()
            all_logits.append(logits)
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def _plot_confusion(cm: np.ndarray, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    thresh = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def evaluate_predictions(y_true, y_pred, *, tag: str = "transformer",
                         out_dir: Path = REPORTS_DIR) -> dict:
    """Compute metrics + plots from integer label arrays. Returns a metrics dict."""
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score,
                                 precision_score, recall_score)

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "tag": tag,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    report = classification_report(
        y_true, y_pred, target_names=LABELS, digits=4, zero_division=0
    )
    print(f"\n===== {tag} =====")
    print(report)
    print(f"accuracy={metrics['accuracy']:.4f} | macro-F1={metrics['f1_macro']:.4f} "
          f"| weighted-F1={metrics['f1_weighted']:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    _plot_confusion(cm, out_dir / f"confusion_matrix_{tag}.png")

    with open(out_dir / f"metrics_{tag}.json", "w", encoding="utf-8") as f:
        json.dump({**metrics, "classification_report": report,
                   "confusion_matrix": cm.tolist()}, f, indent=2)
    return metrics


def evaluate_saved_model(model_dir: Path = None, test_df: pd.DataFrame = None) -> dict:
    """Load the saved best model and evaluate it on the test split."""
    model_dir = model_dir or CONFIG.best_model_path
    if test_df is None:
        from .config import PROCESSED_DIR
        test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    model, tokenizer, meta = HinglishSentimentClassifier.load(model_dir, map_location=DEVICE)
    model.to(DEVICE)

    ds = SentimentDataset(
        texts=test_df["text"].tolist(),
        labels=test_df["label"].map(LABEL2ID).tolist(),
        tokenizer=tokenizer,
        max_length=meta.get("max_length", CONFIG.max_length),
    )
    loader = DataLoader(ds, batch_size=CONFIG.batch_size, shuffle=False)
    logits, y_true = _predict_logits(model, loader)
    y_pred = logits.argmax(dim=-1).numpy()
    return evaluate_predictions(y_true.numpy(), y_pred, tag="transformer")


if __name__ == "__main__":
    evaluate_saved_model()
