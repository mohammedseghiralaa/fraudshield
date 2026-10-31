import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

_model = None
_meta = None


def _load():
    """Load the model and its settings once, then reuse them."""
    global _model, _meta
    if _model is None:
        _model = CatBoostClassifier()
        _model.load_model(str(MODELS_DIR / "catboost_fraud.cbm"))
        with open(MODELS_DIR / "model_meta.json") as f:
            _meta = json.load(f)
    return _model, _meta


def predict_transaction(transaction: dict) -> dict:
    """Score one transaction (a dict with Time, V1..V28, Amount)."""
    model, meta = _load()
    row = dict(transaction)

    # Hour is derived from Time, exactly as in training
    if "Hour" not in row and "Time" in row:
        row["Hour"] = (row["Time"] // 3600) % 24

    missing = [c for c in meta["features"] if c not in row]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    # Same columns, same order as training
    X = pd.DataFrame([row])[meta["features"]]
    proba = float(model.predict_proba(X)[0, 1])

    return {
        "fraud_probability": round(proba, 4),
        "is_fraud": proba >= meta["threshold"],
        "threshold": meta["threshold"],
    }
    