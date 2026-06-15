"""
Dataset loading, cleaning, EDA and splitting for the SentiMix Hinglish dataset.

Pipeline
--------
1. Load `RTT1/SentiMix` from the HuggingFace Hub (with graceful fallbacks for
   differing column names across mirrors of the dataset).
2. Normalise labels -> {Negative, Neutral, Positive}.
3. Clean text via `preprocessing.clean_text`.
4. Drop missing / duplicate / empty rows.
5. Stratified subsample to `CONFIG.subset_size` for fast iteration.
6. Stratified train / val / test split.

Run directly to produce an EDA report + visualisations under `reports/`:

    python -m src.dataset_loader
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import pandas as pd

from .config import CONFIG, LABELS, REPORTS_DIR
from .preprocessing import clean_text, normalize_label

# Candidate column names seen across SentiMix mirrors.
_TEXT_COLS = ["text", "sentence", "tweet", "Text", "clean_text", "content"]
_LABEL_COLS = ["label", "sentiment", "Sentiment", "labels", "class"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _pick_column(df: pd.DataFrame, candidates) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"None of {candidates} found in dataset columns: {list(df.columns)}"
    )


def _parse_conll_sentimix(lines) -> pd.DataFrame:
    """
    Reconstruct tweets from the SemEval-2020 SentiMix CoNLL format.

    The dataset is stored one *token* per row in a single `text` column::

        meta<TAB>4330<TAB>neutral      <- tweet header (id + sentiment)
        nen<TAB>Eng                    <- token<TAB>language-id
        vist<TAB>Eng
        ...
        (blank row)                    <- tweet separator
        meta<TAB>4331<TAB>positive
        ...

    We join the tokens of each tweet back into a sentence and attach the
    sentiment from its `meta` header.
    """
    records, tokens, label = [], [], None

    def _flush():
        if label is not None and tokens:
            records.append({"text": " ".join(tokens), "label": label})

    for raw in lines:
        line = (raw or "").rstrip("\n")
        if not line.strip():               # blank row -> end of tweet
            _flush()
            tokens, label = [], None
            continue
        fields = line.split("\t")
        if fields[0] == "meta" and len(fields) >= 3:
            _flush()                       # close the previous tweet
            tokens, label = [], fields[2]
        else:
            tokens.append(fields[0])
    _flush()                               # final tweet (no trailing blank)

    df = pd.DataFrame(records, columns=["text", "label"])
    df["label"] = df["label"].map(normalize_label)
    return df


def load_raw_dataframe(dataset_name: str = None) -> pd.DataFrame:
    """
    Load the raw dataset from the HuggingFace Hub and return a tidy DataFrame
    with exactly two columns: `text`, `label` (canonical label strings).

    Handles two layouts transparently:
      * tidy `text` + `label`-style columns, and
      * the SentiMix CoNLL token-per-row format (single `text` column).
    """
    from datasets import load_dataset, concatenate_datasets

    dataset_name = dataset_name or CONFIG.dataset_name
    print(f"[loader] Loading '{dataset_name}' from the HuggingFace Hub ...")
    ds = load_dataset(dataset_name)

    # Merge all available splits; we re-split ourselves for a clean stratified setup.
    parts = [ds[s] for s in ds.keys()]
    merged = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    df = merged.to_pandas()

    # CoNLL token-per-row layout: a single text column and no usable label column.
    has_label = any(c in df.columns for c in _LABEL_COLS)
    if not has_label and "text" in df.columns:
        print("[loader] Detected SentiMix CoNLL format - reconstructing tweets ...")
        df = _parse_conll_sentimix(df["text"].tolist())
        print(f"[loader] Reconstructed {len(df):,} tweets from token rows.")
        return df

    text_col = _pick_column(df, _TEXT_COLS)
    label_col = _pick_column(df, _LABEL_COLS)
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})

    df["label"] = df["label"].map(normalize_label)
    print(f"[loader] Loaded {len(df):,} raw rows.")
    return df


def load_local_csv(path: str | Path) -> pd.DataFrame:
    """Fallback loader for a local CSV with `text,label` columns."""
    df = pd.read_csv(path)
    text_col = _pick_column(df, _TEXT_COLS)
    label_col = _pick_column(df, _LABEL_COLS)
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df["label"] = df["label"].map(normalize_label)
    return df


# --------------------------------------------------------------------------- #
# Cleaning / dedup
# --------------------------------------------------------------------------- #
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning, drop missing/empty/duplicate rows, report stats."""
    n0 = len(df)
    stats = {"initial_rows": n0}

    # Missing values
    stats["missing_text"] = int(df["text"].isna().sum())
    stats["missing_label"] = int(df["label"].isna().sum())
    df = df.dropna(subset=["text", "label"]).copy()

    # Clean text
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0]

    # Keep only canonical labels
    df = df[df["label"].isin(LABELS)]

    # Duplicates
    dups = int(df.duplicated(subset=["text"]).sum())
    stats["duplicate_rows"] = dups
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    stats["final_rows"] = len(df)
    stats["dropped_rows"] = n0 - len(df)
    print(f"[loader] Cleaning: {n0:,} -> {len(df):,} rows "
          f"(dropped {stats['dropped_rows']:,}; {dups:,} dupes).")
    return df, stats


# --------------------------------------------------------------------------- #
# EDA
# --------------------------------------------------------------------------- #
def run_eda(df: pd.DataFrame, stats: dict, out_dir: Path = REPORTS_DIR) -> dict:
    """Compute EDA stats and write visualisations (class dist + text length)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["n_words"] = df["text"].str.split().map(len)
    df["n_chars"] = df["text"].str.len()

    class_counts = df["label"].value_counts().reindex(LABELS).fillna(0).astype(int)
    eda = {
        **stats,
        "class_distribution": class_counts.to_dict(),
        "word_length": {
            "mean": float(df["n_words"].mean()),
            "median": float(df["n_words"].median()),
            "p95": float(df["n_words"].quantile(0.95)),
            "max": int(df["n_words"].max()),
        },
    }

    # --- Plot 1: class distribution ---
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#e74c3c", "#95a5a6", "#2ecc71"]
    ax.bar(class_counts.index, class_counts.values, color=colors)
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    for i, v in enumerate(class_counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "class_distribution.png", dpi=120)
    plt.close(fig)

    # --- Plot 2: text length distribution ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["n_words"].clip(upper=60), bins=40, color="#3498db", alpha=0.85)
    ax.axvline(df["n_words"].median(), color="red", linestyle="--",
               label=f"median={df['n_words'].median():.0f}")
    ax.set_title("Text Length Distribution (words)")
    ax.set_xlabel("Words per sample")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "text_length_distribution.png", dpi=120)
    plt.close(fig)

    with open(out_dir / "eda_report.json", "w", encoding="utf-8") as f:
        json.dump(eda, f, indent=2)

    print(f"[loader] EDA written to {out_dir} (class_distribution.png, "
          f"text_length_distribution.png, eda_report.json)")
    return eda


# --------------------------------------------------------------------------- #
# Subsample + split
# --------------------------------------------------------------------------- #
def stratified_subsample(df: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Take a class-stratified subsample of up to `size` rows."""
    if size <= 0 or size >= len(df):
        return df.reset_index(drop=True)
    frac = size / len(df)
    # groupby().sample() keeps all columns and is stable across pandas versions
    # (unlike groupby().apply(), whose grouping-column handling changed in 3.0).
    sub = df.groupby("label", group_keys=False).sample(frac=frac, random_state=seed)
    return sub.reset_index(drop=True)


def split_dataframe(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / val / test split based on CONFIG ratios."""
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(
        df,
        test_size=CONFIG.test_size + CONFIG.val_size,
        stratify=df["label"],
        random_state=CONFIG.random_seed,
    )
    rel_val = CONFIG.val_size / (CONFIG.test_size + CONFIG.val_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - rel_val,
        stratify=temp_df["label"],
        random_state=CONFIG.random_seed,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_datasets(
    save: bool = True, run_eda_report: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """End-to-end: load -> clean -> EDA -> subsample -> split. Returns splits."""
    raw = load_raw_dataframe()
    clean, stats = clean_dataframe(raw)
    if run_eda_report:
        run_eda(clean, stats)
    sub = stratified_subsample(clean, CONFIG.subset_size, CONFIG.random_seed)
    train_df, val_df, test_df = split_dataframe(sub)

    print(f"[loader] Split -> train={len(train_df):,} | "
          f"val={len(val_df):,} | test={len(test_df):,}")

    if save:
        from .config import PROCESSED_DIR
        train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
        val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
        test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
        print(f"[loader] Saved splits to {PROCESSED_DIR}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build & analyse the dataset.")
    parser.add_argument("--no-save", action="store_true", help="Do not write CSV splits.")
    parser.add_argument("--no-eda", action="store_true", help="Skip EDA visualisations.")
    args = parser.parse_args()
    build_datasets(save=not args.no_save, run_eda_report=not args.no_eda)
