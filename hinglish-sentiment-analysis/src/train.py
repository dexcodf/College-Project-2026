"""
Training pipeline for the Hinglish sentiment transformer.

Custom PyTorch loop (instead of HF Trainer) so we can:
  * track per-epoch train/val loss + accuracy for the requested curves,
  * support CUDA / Apple MPS / CPU transparently,
  * apply early stopping and save the best checkpoint.

Usage:
    python -m src.train                 # uses CONFIG defaults
    EPOCHS=3 BATCH_SIZE=16 python -m src.train
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import CONFIG, DEVICE, LABEL2ID, REPORTS_DIR
from .model import HinglishSentimentClassifier, SentimentDataset, load_tokenizer


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def _load_splits():
    """Load processed CSV splits, building them on the fly if missing."""
    from .config import PROCESSED_DIR

    files = [PROCESSED_DIR / f"{s}.csv" for s in ("train", "val", "test")]
    if not all(f.exists() for f in files):
        print("[train] Processed splits not found - building them now ...")
        from .dataset_loader import build_datasets
        return build_datasets(save=True)
    return tuple(pd.read_csv(f) for f in files)


def _make_loader(df: pd.DataFrame, tokenizer, shuffle: bool) -> DataLoader:
    ds = SentimentDataset(
        texts=df["text"].tolist(),
        labels=df["label"].map(LABEL2ID).tolist(),
        tokenizer=tokenizer,
        max_length=CONFIG.max_length,
    )
    return DataLoader(ds, batch_size=CONFIG.batch_size, shuffle=shuffle)


# --------------------------------------------------------------------------- #
# Train / eval loops
# --------------------------------------------------------------------------- #
def _run_epoch(model, loader, optimizer=None, scheduler=None, train: bool = True):
    model.train() if train else model.eval()
    total_loss, n_correct, n_seen = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch in tqdm(loader, desc="train" if train else "eval", leave=False):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch)
            loss, logits = out["loss"], out["logits"]

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(dim=-1)
            n_correct += (preds == batch["labels"]).sum().item()
            n_seen += batch["labels"].size(0)

    return total_loss / n_seen, n_correct / n_seen


def _plot_curves(history: dict, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], "o-", label="train")
    axes[0].plot(epochs, history["val_loss"], "o-", label="val")
    axes[0].set_title("Loss Curve"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss"); axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], "o-", label="train")
    axes[1].plot(epochs, history["val_acc"], "o-", label="val")
    axes[1].set_title("Accuracy Curve"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy"); axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main training entry point
# --------------------------------------------------------------------------- #
def train(run_eval: bool = True):
    from transformers import get_linear_schedule_with_warmup

    print(f"[train] Device: {DEVICE} | Model: {CONFIG.model_name}")
    torch.manual_seed(CONFIG.random_seed)
    np.random.seed(CONFIG.random_seed)

    train_df, val_df, test_df = _load_splits()
    tokenizer = load_tokenizer()

    train_loader = _make_loader(train_df, tokenizer, shuffle=True)
    val_loader = _make_loader(val_df, tokenizer, shuffle=False)

    model = HinglishSentimentClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay
    )
    total_steps = len(train_loader) * CONFIG.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(CONFIG.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, patience = -1.0, 0
    t0 = time.time()

    for epoch in range(1, CONFIG.epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, optimizer, scheduler, train=True)
        va_loss, va_acc = _run_epoch(model, val_loader, train=False)

        history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc); history["val_acc"].append(va_acc)
        print(f"[train] Epoch {epoch}/{CONFIG.epochs} | "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc, patience = va_acc, 0
            model.save(CONFIG.best_model_path, tokenizer=tokenizer)
            print(f"[train]   -> new best (val_acc={va_acc:.4f}) saved to "
                  f"{CONFIG.best_model_path}")
        else:
            patience += 1
            if patience >= CONFIG.early_stopping_patience:
                print(f"[train] Early stopping at epoch {epoch}.")
                break

    elapsed = time.time() - t0
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _plot_curves(history, REPORTS_DIR)
    with open(REPORTS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, "best_val_acc": best_val_acc,
                   "elapsed_sec": elapsed}, f, indent=2)
    print(f"[train] Done in {elapsed/60:.1f} min. Best val_acc={best_val_acc:.4f}")
    print(f"[train] Curves -> {REPORTS_DIR / 'training_curves.png'}")

    if run_eval:
        from .evaluate import evaluate_saved_model
        evaluate_saved_model(test_df=test_df)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Hinglish sentiment model.")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip test-set evaluation after training.")
    args = parser.parse_args()
    train(run_eval=not args.no_eval)
