# Titanic Survival Prediction

Predicts whether a passenger survived the Titanic disaster from demographic and
ticket information.

**Dataset:** [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

## Project structure

```
.
├── data/                  # train.csv, test.csv
├── src/
│   └── train.py           # full training + prediction pipeline
├── models/
│   └── titanic_model.joblib   # fitted model (saved by train.py)
├── outputs/
│   └── submission.csv     # Kaggle-format predictions on the test set
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/train.py
```

This will:

1. Load `data/train.csv` and `data/test.csv`.
2. Engineer features (passenger Title, FamilySize, IsAlone, Deck, HasCabin, ...).
3. Compare three models with 5-fold stratified cross-validation.
4. Refit the best model on the full training set.
5. Save the model to `models/` and write `outputs/submission.csv`.

## Approach

**Feature engineering**

| Feature      | Description                                              |
|--------------|----------------------------------------------------------|
| `Title`      | Honorific parsed from the name (Mr, Mrs, Miss, Master, Rare) |
| `FamilySize` | `SibSp + Parch + 1`                                      |
| `IsAlone`    | 1 when travelling alone                                  |
| `Deck`       | First letter of the cabin code (`Unknown` if missing)   |
| `HasCabin`   | Whether a cabin was recorded                             |

**Preprocessing** — median imputation + standard scaling for numerics, mode
imputation + one-hot encoding for categoricals (all inside an sklearn
`Pipeline`, so there is no train/test leakage).

**Models compared** — Logistic Regression, Random Forest, Gradient Boosting.

## Results

5-fold stratified cross-validation accuracy:

| Model               | CV Accuracy |
|---------------------|-------------|
| Logistic Regression | 0.8305      |
| Random Forest       | 0.8339      |
| **Gradient Boosting** | **0.8417** |

The best model (Gradient Boosting) is retrained on all 891 training rows and
used to generate `outputs/submission.csv`.
