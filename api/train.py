"""
Run during Render build to generate model artifacts.
Called via build command before uvicorn starts.
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

SEED = 42
N    = 2000
np.random.seed(SEED)

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

WEIGHTS = {
    "tx_regularity_score":   0.25,
    "bill_payment_rate":     0.25,
    "income_stability":      0.20,
    "has_savings_behaviour": 0.10,
    "merchant_diversity":    0.08,
    "avg_monthly_inflow":    0.08,
    "account_age_months":    0.04,
}

FEATURE_COLS = list(WEIGHTS.keys())

def generate():
    tx_regularity_score   = np.random.beta(3, 4, N) * 100
    bill_payment_rate     = np.random.beta(3, 3, N) * 100
    income_stability      = np.random.beta(2, 4, N) * 100
    has_savings_behaviour = np.random.binomial(1, 0.35, N).astype(float)
    merchant_diversity    = np.random.poisson(4, N).clip(0, 15).astype(float)
    avg_monthly_inflow    = np.random.exponential(35, N).clip(5, 500)
    account_age_months    = np.random.randint(1, 49, N).astype(float)

    df = pd.DataFrame({
        "tx_regularity_score":   tx_regularity_score,
        "bill_payment_rate":     bill_payment_rate,
        "income_stability":      income_stability,
        "has_savings_behaviour": has_savings_behaviour,
        "merchant_diversity":    merchant_diversity,
        "avg_monthly_inflow":    avg_monthly_inflow,
        "account_age_months":    account_age_months,
    })

    merchant_norm = (df["merchant_diversity"]   / 15)  * 100
    inflow_norm   = (df["avg_monthly_inflow"]   / 500) * 100
    age_norm      = (df["account_age_months"]   / 48)  * 100

    df["credit_score"] = (
        WEIGHTS["tx_regularity_score"]   * df["tx_regularity_score"]   +
        WEIGHTS["bill_payment_rate"]     * df["bill_payment_rate"]      +
        WEIGHTS["income_stability"]      * df["income_stability"]       +
        WEIGHTS["has_savings_behaviour"] * df["has_savings_behaviour"]  * 100 +
        WEIGHTS["merchant_diversity"]    * merchant_norm                +
        WEIGHTS["avg_monthly_inflow"]    * inflow_norm                  +
        WEIGHTS["account_age_months"]    * age_norm
    ).clip(0, 100)

    return df

def train(df):
    X = df[FEATURE_COLS]
    y = df["credit_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=1.0))
    ])
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, MODELS_DIR / "creditpulse_model.pkl")

    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    print(f"Model trained and saved → {MODELS_DIR}")

if __name__ == "__main__":
    print("Generating data...")
    df = generate()
    print("Training model...")
    train(df)
    print("Done.")