"""
Titanic Survival Prediction - Training Pipeline
================================================

Trains and evaluates classifiers to predict passenger survival on the
Titanic dataset (Kaggle: "Titanic - Machine Learning from Disaster").

Workflow:
    1. Load train/test CSVs.
    2. Engineer features (Title, FamilySize, IsAlone, Deck, ...).
    3. Build a preprocessing + model Pipeline.
    4. Compare several models with stratified cross-validation.
    5. Refit the best model on all training data.
    6. Save the fitted model and a Kaggle-format submission file.

Run:
    python src/train.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive new features from the raw Titanic columns.

    These transformations are stateless (no fitting on the data), so they can
    be applied identically to the train and test sets.
    """
    df = df.copy()

    # Title extracted from the passenger name, e.g. "Braund, Mr. Owen" -> "Mr".
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    rare_titles = {
        "Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev",
        "Sir", "Jonkheer", "Dona",
    }
    df["Title"] = df["Title"].replace(list(rare_titles), "Rare")
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    # Family structure.
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Deck letter from the cabin code; "Unknown" when missing.
    df["Deck"] = df["Cabin"].str[0]
    df["Deck"] = df["Deck"].fillna("Unknown")

    # Whether the ticket fare / cabin info was recorded at all.
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    return df


# Columns fed to the model after engineering.
NUMERIC_FEATURES = ["Age", "Fare", "FamilySize", "SibSp", "Parch", "IsAlone", "HasCabin"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked", "Title", "Deck"]


def build_preprocessor() -> ColumnTransformer:
    """Impute, scale numerics and one-hot encode categoricals."""
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def candidate_models() -> dict:
    """Return the set of models to compare."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }


def main() -> None:
    print("=" * 70)
    print("Titanic Survival Prediction - Training")
    print("=" * 70)

    # ----------------------------------------------------------------- Load
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    print(f"Train: {train.shape}, Test: {test.shape}")

    train = engineer_features(train)
    test = engineer_features(test)

    X = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = train["Survived"]

    preprocessor = build_preprocessor()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ------------------------------------------------------- Compare models
    print("\n5-fold stratified cross-validation (accuracy):")
    print("-" * 70)
    results = {}
    for name, model in candidate_models().items():
        pipe = Pipeline(steps=[("prep", preprocessor), ("clf", model)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        results[name] = scores.mean()
        print(f"  {name:<20s} {scores.mean():.4f}  (+/- {scores.std():.4f})")

    best_name = max(results, key=results.get)
    print(f"\nBest model: {best_name}  (CV accuracy = {results[best_name]:.4f})")

    # ------------------------------------------------- Refit on all train
    best_pipe = Pipeline(
        steps=[("prep", build_preprocessor()), ("clf", candidate_models()[best_name])]
    )
    best_pipe.fit(X, y)

    # Training-set report (for reference; CV above is the honest estimate).
    train_pred = best_pipe.predict(X)
    print("\nTraining-set performance (optimistic):")
    print(f"  Accuracy: {accuracy_score(y, train_pred):.4f}")
    print("  Confusion matrix:")
    print(confusion_matrix(y, train_pred))
    print("  Classification report:")
    print(classification_report(y, train_pred, target_names=["Died", "Survived"]))

    # ----------------------------------------------------------- Save model
    model_path = MODEL_DIR / "titanic_model.joblib"
    joblib.dump(best_pipe, model_path)
    print(f"Saved fitted model -> {model_path.relative_to(ROOT)}")

    # ------------------------------------------------- Predict on test set
    X_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    test_pred = best_pipe.predict(X_test)
    submission = pd.DataFrame(
        {"PassengerId": test["PassengerId"], "Survived": test_pred.astype(int)}
    )
    sub_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"Saved submission ({len(submission)} rows) -> {sub_path.relative_to(ROOT)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
