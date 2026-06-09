"""
Titanic Survival Prediction - Streamlit Dashboard
=================================================

An interactive dashboard for the Titanic project with four sections:

    1. Overview      - dataset summary and key metrics
    2. Exploration   - survival breakdowns and distributions (Altair charts)
    3. Model         - honest (out-of-fold) performance + feature importance
    4. Predict       - live survival prediction from passenger inputs

Run:
    streamlit run src/dashboard.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt
import joblib
import streamlit as st

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------------------------------- #
# Config & paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "titanic_model.joblib"
RANDOM_STATE = 42

NUMERIC_FEATURES = ["Age", "Fare", "FamilySize", "SibSp", "Parch", "IsAlone", "HasCabin"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked", "Title", "Deck"]

st.set_page_config(
    page_title="Titanic Survival Dashboard",
    page_icon="🚢",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Feature engineering (kept in sync with src/train.py)
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    rare = ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev",
            "Sir", "Jonkheer", "Dona"]
    df["Title"] = df["Title"].replace(rare, "Rare")
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Deck"] = df["Cabin"].str[0].fillna("Unknown")
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    return df


# --------------------------------------------------------------------------- #
# Cached loaders / computations
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, test


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Evaluating model (out-of-fold)...")
def evaluate(_model, X: pd.DataFrame, y: pd.Series):
    """Honest metrics using out-of-fold predictions on the training data."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    estimator = clone(_model)
    oof_pred = cross_val_predict(estimator, X, y, cv=cv, n_jobs=-1)
    cv_acc = cross_val_score(estimator, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return {
        "oof_pred": oof_pred,
        "cv_acc_mean": float(cv_acc.mean()),
        "cv_acc_std": float(cv_acc.std()),
        "accuracy": accuracy_score(y, oof_pred),
        "precision": precision_score(y, oof_pred),
        "recall": recall_score(y, oof_pred),
        "f1": f1_score(y, oof_pred),
        "cm": confusion_matrix(y, oof_pred),
        "report": classification_report(
            y, oof_pred, target_names=["Died", "Survived"], output_dict=True
        ),
    }


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
train_raw, test_raw = load_data()
model = load_model()
train = engineer_features(train_raw)

st.title("🚢 Titanic Survival Prediction Dashboard")
st.caption(
    "Kaggle: *Titanic - Machine Learning from Disaster* · "
    "Gradient Boosting classifier with engineered features."
)

if model is None:
    st.error("Model file not found. Run `python src/train.py` first to create "
             "`models/titanic_model.joblib`.")
    st.stop()

tab_overview, tab_eda, tab_model, tab_predict = st.tabs(
    ["📊 Overview", "🔎 Exploration", "🎯 Model", "🔮 Predict"]
)

# --------------------------------------------------------------------------- #
# Tab 1: Overview
# --------------------------------------------------------------------------- #
with tab_overview:
    st.subheader("Dataset at a glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Passengers (train)", f"{len(train_raw):,}")
    c2.metric("Passengers (test)", f"{len(test_raw):,}")
    c3.metric("Survival rate", f"{train_raw['Survived'].mean() * 100:.1f}%")
    c4.metric("Features used", len(NUMERIC_FEATURES + CATEGORICAL_FEATURES))

    st.markdown("#### Sample of the training data")
    st.dataframe(train_raw.head(10), use_container_width=True)

    st.markdown("#### Missing values")
    miss = train_raw.isnull().sum()
    miss = miss[miss > 0].rename("missing").to_frame()
    miss["% of rows"] = (miss["missing"] / len(train_raw) * 100).round(1)
    st.dataframe(miss, use_container_width=True)

# --------------------------------------------------------------------------- #
# Tab 2: Exploration
# --------------------------------------------------------------------------- #
with tab_eda:
    st.subheader("Who survived?")

    def rate_chart(col: str, title: str):
        g = (train.groupby(col)["Survived"].mean().reset_index())
        g["Survived"] = (g["Survived"] * 100).round(1)
        return (
            alt.Chart(g)
            .mark_bar()
            .encode(
                x=alt.X(f"{col}:N", title=title),
                y=alt.Y("Survived:Q", title="Survival rate (%)"),
                color=alt.Color(f"{col}:N", legend=None),
                tooltip=[col, "Survived"],
            )
            .properties(height=300)
        )

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**By sex**")
        st.altair_chart(rate_chart("Sex", "Sex"), use_container_width=True)
    with r2:
        st.markdown("**By passenger class**")
        st.altair_chart(rate_chart("Pclass", "Pclass"), use_container_width=True)

    r3, r4 = st.columns(2)
    with r3:
        st.markdown("**By title**")
        st.altair_chart(rate_chart("Title", "Title"), use_container_width=True)
    with r4:
        st.markdown("**By family size**")
        st.altair_chart(rate_chart("FamilySize", "Family size"), use_container_width=True)

    st.markdown("**Age distribution by outcome**")
    age_df = train.dropna(subset=["Age"]).copy()
    age_df["Outcome"] = age_df["Survived"].map({0: "Died", 1: "Survived"})
    hist = (
        alt.Chart(age_df)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("Age:Q", bin=alt.Bin(maxbins=30), title="Age"),
            y=alt.Y("count()", title="Passengers", stack=None),
            color=alt.Color("Outcome:N",
                            scale=alt.Scale(domain=["Died", "Survived"],
                                            range=["#ef4444", "#22c55e"])),
            tooltip=["Outcome", "count()"],
        )
        .properties(height=320)
    )
    st.altair_chart(hist, use_container_width=True)

# --------------------------------------------------------------------------- #
# Tab 3: Model performance
# --------------------------------------------------------------------------- #
with tab_model:
    st.subheader("Model performance (5-fold out-of-fold)")
    X = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = train["Survived"]
    ev = evaluate(model, X, y)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CV Accuracy", f"{ev['cv_acc_mean']:.3f}", f"± {ev['cv_acc_std']:.3f}")
    m2.metric("Precision", f"{ev['precision']:.3f}")
    m3.metric("Recall", f"{ev['recall']:.3f}")
    m4.metric("F1 score", f"{ev['f1']:.3f}")

    cleft, cright = st.columns(2)
    with cleft:
        st.markdown("#### Confusion matrix")
        cm = ev["cm"]
        cm_df = pd.DataFrame(
            cm, index=["Actual: Died", "Actual: Survived"],
            columns=["Pred: Died", "Pred: Survived"],
        )
        cm_long = cm_df.reset_index().melt(id_vars="index")
        cm_long.columns = ["Actual", "Predicted", "Count"]
        heat = (
            alt.Chart(cm_long)
            .mark_rect()
            .encode(
                x=alt.X("Predicted:N"),
                y=alt.Y("Actual:N"),
                color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")),
                tooltip=["Actual", "Predicted", "Count"],
            )
            .properties(height=260)
        )
        text = heat.mark_text(baseline="middle", fontSize=18).encode(
            text="Count:Q",
            color=alt.condition(alt.datum.Count > cm.max() / 2,
                                alt.value("white"), alt.value("black")),
        )
        st.altair_chart(heat + text, use_container_width=True)

    with cright:
        st.markdown("#### Feature importance")
        prep = model.named_steps["prep"]
        clf = model.named_steps["clf"]
        feat_names = prep.get_feature_names_out()
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
        else:  # linear model fallback
            imp = np.abs(np.ravel(clf.coef_))
        fi = (
            pd.DataFrame({"feature": feat_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .head(12)
        )
        fi["feature"] = fi["feature"].str.replace("num__|cat__", "", regex=True)
        bar = (
            alt.Chart(fi)
            .mark_bar(color="#38bdf8")
            .encode(
                x=alt.X("importance:Q", title="Importance"),
                y=alt.Y("feature:N", sort="-x", title=None),
                tooltip=["feature", "importance"],
            )
            .properties(height=320)
        )
        st.altair_chart(bar, use_container_width=True)

    st.markdown("#### Classification report")
    st.dataframe(pd.DataFrame(ev["report"]).T.round(3), use_container_width=True)

# --------------------------------------------------------------------------- #
# Tab 4: Live prediction
# --------------------------------------------------------------------------- #
with tab_predict:
    st.subheader("Predict survival for a passenger")

    with st.form("predict_form"):
        a, b, c = st.columns(3)
        with a:
            pclass = st.selectbox("Passenger class", [1, 2, 3], index=2,
                                  format_func=lambda x: f"{x} ({['1st','2nd','3rd'][x-1]})")
            sex = st.selectbox("Sex", ["male", "female"])
            title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])
        with b:
            age = st.slider("Age", 0, 80, 30)
            fare = st.slider("Fare (£)", 0.0, 300.0, 32.0, step=0.5)
            embarked = st.selectbox("Embarked",
                                    ["S", "C", "Q"],
                                    format_func=lambda x: {"S": "Southampton",
                                                           "C": "Cherbourg",
                                                           "Q": "Queenstown"}[x])
        with c:
            sibsp = st.number_input("Siblings/Spouses aboard", 0, 10, 0)
            parch = st.number_input("Parents/Children aboard", 0, 10, 0)
            deck = st.selectbox("Cabin deck",
                                ["Unknown", "A", "B", "C", "D", "E", "F", "G", "T"])
        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        family_size = sibsp + parch + 1
        row = pd.DataFrame([{
            "Age": age, "Fare": fare, "FamilySize": family_size,
            "SibSp": sibsp, "Parch": parch,
            "IsAlone": 1 if family_size == 1 else 0,
            "HasCabin": 0 if deck == "Unknown" else 1,
            "Pclass": pclass, "Sex": sex, "Embarked": embarked,
            "Title": title, "Deck": deck,
        }], columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)

        pred = int(model.predict(row)[0])
        proba = float(model.predict_proba(row)[0][1])

        if pred == 1:
            st.success(f"### ✅ Likely **SURVIVED**")
        else:
            st.error(f"### ❌ Likely **did not survive**")
        st.progress(proba, text=f"Survival probability: {proba * 100:.1f}%")
