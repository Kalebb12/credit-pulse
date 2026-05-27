import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent.parent
MODEL_PATH  = BASE_DIR / "models" / "creditpulse_model.pkl"
COLS_PATH   = BASE_DIR / "models" / "feature_columns.json"

pipeline = joblib.load(MODEL_PATH)

with open(COLS_PATH) as f:
    FEATURE_COLS = json.load(f)

model        = pipeline.named_steps["model"]
coefficients = model.coef_
importance   = np.abs(coefficients) / np.abs(coefficients).sum()

FEATURE_WEIGHTS = dict(zip(FEATURE_COLS, importance))

def get_tier(score: float) -> tuple[str, bool, int]:
    """Returns (tier_name, eligible, loan_ceiling_naira)"""
    if score < 40:
        return "Thin File", False, 0
    elif score < 60:
        # Standard: ₦10,000 → ₦50,000
        ceiling = int(round(10000 + ((score - 40) / 20) * 40000, -3))
        return "Standard", True, ceiling
    else:
        # Prime: ₦50,000 → ₦200,000
        ceiling = int(round(50000 + ((score - 60) / 40) * 150000, -3))
        return "Prime", True, min(ceiling, 200000)


def predict(data: dict) -> dict:
    X = pd.DataFrame([data])[FEATURE_COLS]

    raw_score = float(pipeline.predict(X)[0])
    score     = int(round(np.clip(raw_score, 0, 100)))

    tier, eligible, ceiling = get_tier(score)

    contributions = {
        col: round(float(FEATURE_WEIGHTS[col]) * 100, 2)
        for col in FEATURE_COLS
    }

    return {
        "credit_score":          score,
        "tier":                  tier,
        "eligible":              eligible,
        "loan_ceiling_naira":    ceiling,
        "feature_contributions": contributions,
    }