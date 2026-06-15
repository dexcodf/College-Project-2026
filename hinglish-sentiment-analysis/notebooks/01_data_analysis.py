"""
Standalone EDA script (mirrors what a notebook would contain).

Run as a script:
    python notebooks/01_data_analysis.py

or open in Jupyter / Colab and execute cell by cell (cells are delimited by
`# %%` markers, recognised by VS Code and Jupytext).
"""
# %%
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# %% Load + clean
from src.dataset_loader import clean_dataframe, load_raw_dataframe, run_eda  # noqa: E402

raw = load_raw_dataframe()
print(raw.head())
print("Raw shape:", raw.shape)

# %% Missing values & duplicates
print("Missing per column:\n", raw.isna().sum())
print("Duplicate texts:", raw.duplicated(subset=['text']).sum())

# %% Clean + EDA report (writes plots to reports/)
clean, stats = clean_dataframe(raw)
eda = run_eda(clean, stats)
print("\nEDA summary:")
import json  # noqa: E402
print(json.dumps(eda, indent=2))

# %% Class distribution
print("\nClass distribution:")
print(clean['label'].value_counts())

# %% Sample texts per class
for label in ['Positive', 'Neutral', 'Negative']:
    print(f"\n--- {label} samples ---")
    print(clean[clean['label'] == label]['text'].head(3).to_list())
