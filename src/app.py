"""
Titanic Survival Prediction - Manual Testing Web App
====================================================

A small Flask dev server for manually testing the trained model. It loads the
saved pipeline (models/titanic_model.joblib) and serves a form where you can
enter passenger details and get a live survival prediction + probability.

Run:
    python src/app.py
Then open http://127.0.0.1:5000 in a browser.
"""

from pathlib import Path

import pandas as pd
import joblib
from flask import Flask, request, render_template_string

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "titanic_model.joblib"

# Columns the pipeline expects (must match src/train.py).
NUMERIC_FEATURES = ["Age", "Fare", "FamilySize", "SibSp", "Parch", "IsAlone", "HasCabin"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked", "Title", "Deck"]

app = Flask(__name__)

if not MODEL_PATH.exists():
    raise SystemExit(
        f"Model not found at {MODEL_PATH}.\nRun `python src/train.py` first to create it."
    )
model = joblib.load(MODEL_PATH)


def build_features(form) -> pd.DataFrame:
    """Turn raw form inputs into the single-row frame the pipeline expects."""
    pclass = int(form.get("Pclass", 3))
    sex = form.get("Sex", "male")
    age = float(form.get("Age", 30) or 30)
    sibsp = int(form.get("SibSp", 0) or 0)
    parch = int(form.get("Parch", 0) or 0)
    fare = float(form.get("Fare", 32) or 32)
    embarked = form.get("Embarked", "S")
    title = form.get("Title", "Mr")
    deck = form.get("Deck", "Unknown")

    family_size = sibsp + parch + 1
    row = {
        "Age": age,
        "Fare": fare,
        "FamilySize": family_size,
        "SibSp": sibsp,
        "Parch": parch,
        "IsAlone": 1 if family_size == 1 else 0,
        "HasCabin": 0 if deck == "Unknown" else 1,
        "Pclass": pclass,
        "Sex": sex,
        "Embarked": embarked,
        "Title": title,
        "Deck": deck,
    }
    return pd.DataFrame([row], columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)


PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Titanic Survival Predictor</title>
  <style>
    :root { --bg:#0f172a; --card:#1e293b; --accent:#38bdf8; --text:#e2e8f0; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: system-ui, Segoe UI, Roboto, sans-serif;
           background: var(--bg); color: var(--text); padding: 2rem; }
    .wrap { max-width: 640px; margin: 0 auto; }
    h1 { font-size: 1.6rem; margin-bottom: .25rem; }
    p.sub { color:#94a3b8; margin-top:0; }
    form { background: var(--card); padding: 1.5rem; border-radius: 14px;
           display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
    label { display:flex; flex-direction:column; font-size:.85rem; gap:.3rem; }
    input, select { padding:.55rem .6rem; border-radius:8px; border:1px solid #334155;
                    background:#0b1220; color:var(--text); font-size:.95rem; }
    button { grid-column: 1 / -1; padding:.8rem; border:0; border-radius:10px;
             background: var(--accent); color:#04293a; font-weight:700;
             font-size:1rem; cursor:pointer; }
    button:hover { filter: brightness(1.08); }
    .result { grid-column:1/-1; margin-top:.5rem; padding:1.1rem; border-radius:10px;
              text-align:center; font-size:1.15rem; }
    .survived { background: rgba(34,197,94,.15); border:1px solid #22c55e; }
    .died { background: rgba(239,68,68,.15); border:1px solid #ef4444; }
    .prob { font-size:.9rem; color:#94a3b8; margin-top:.4rem; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>🚢 Titanic Survival Predictor</h1>
    <p class="sub">Gradient Boosting model &middot; CV accuracy ~0.84. Enter passenger details below.</p>
    <form method="post" action="/predict">
      <label>Passenger Class
        <select name="Pclass">
          {% for v,t in [(1,'1st'),(2,'2nd'),(3,'3rd')] %}
          <option value="{{v}}" {{'selected' if f.get('Pclass')==v|string else ''}}>{{t}}</option>
          {% endfor %}
        </select>
      </label>
      <label>Sex
        <select name="Sex">
          {% for v in ['male','female'] %}
          <option value="{{v}}" {{'selected' if f.get('Sex')==v else ''}}>{{v}}</option>
          {% endfor %}
        </select>
      </label>
      <label>Age <input type="number" step="1" min="0" max="100" name="Age" value="{{f.get('Age','30')}}"></label>
      <label>Fare (£) <input type="number" step="0.1" min="0" name="Fare" value="{{f.get('Fare','32')}}"></label>
      <label>Siblings/Spouses aboard <input type="number" step="1" min="0" name="SibSp" value="{{f.get('SibSp','0')}}"></label>
      <label>Parents/Children aboard <input type="number" step="1" min="0" name="Parch" value="{{f.get('Parch','0')}}"></label>
      <label>Embarked
        <select name="Embarked">
          {% for v,t in [('S','Southampton'),('C','Cherbourg'),('Q','Queenstown')] %}
          <option value="{{v}}" {{'selected' if f.get('Embarked')==v else ''}}>{{t}}</option>
          {% endfor %}
        </select>
      </label>
      <label>Title
        <select name="Title">
          {% for v in ['Mr','Mrs','Miss','Master','Rare'] %}
          <option value="{{v}}" {{'selected' if f.get('Title')==v else ''}}>{{v}}</option>
          {% endfor %}
        </select>
      </label>
      <label>Cabin Deck
        <select name="Deck">
          {% for v in ['Unknown','A','B','C','D','E','F','G','T'] %}
          <option value="{{v}}" {{'selected' if f.get('Deck')==v else ''}}>{{v}}</option>
          {% endfor %}
        </select>
      </label>
      <button type="submit">Predict survival</button>
      {% if prediction is not none %}
      <div class="result {{ 'survived' if prediction==1 else 'died' }}">
        {{ '✅ Survived' if prediction==1 else '❌ Did not survive' }}
        <div class="prob">Survival probability: {{ '%.1f'|format(proba*100) }}%</div>
      </div>
      {% endif %}
    </form>
  </div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(PAGE, f={}, prediction=None, proba=None)


@app.route("/predict", methods=["POST"])
def predict():
    X = build_features(request.form)
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    return render_template_string(PAGE, f=request.form, prediction=pred, proba=proba)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON endpoint: POST passenger fields, get {prediction, probability}."""
    data = request.get_json(force=True, silent=True) or {}
    X = build_features(data)
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    return {"prediction": pred, "survived": bool(pred), "probability": round(proba, 4)}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
