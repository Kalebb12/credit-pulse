# CreditPulse

A hackathon backend prototype for OPay to score users with limited or no formal credit history using behavioural transaction data.

## Problem

Millions of Nigerians are excluded from loans because they lack formal credit history. OPay already has rich behavioural signals from transactions, airtime top-ups, transfers, and bill payments.

## Solution

CreditPulse builds an alternative credit scoring model using behavioural data as proxy signals for creditworthiness. The backend computes:

- a risk score out of 100
- a borrower tier
- loan eligibility
- a recommended loan ceiling in Naira

This is aligned with OPay's financial inclusion mission and offers a credible AI/ML-based credit solution without requiring bank statements.

## What’s implemented

- Backend API in `api/main.py`
- Scoring model logic in `api/model.py`
- Request/response schema in `api/schema.py`
- Sample data and feature definitions in `data/` and `models/`
- Model analysis and plots in `notebooks/03_plots.ipynb`

## Tech stack

- Python
- FastAPI
- scikit-learn / joblib
- pandas / numpy
- Pydantic

## How to run

1. Create a Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r api/requirements.txt
```

2. Start the API from the `api/` folder

```bash
cd api
uvicorn main:app --reload
```

3. The endpoint is available at:

```text
http://127.0.0.1:8000/score
```

## API input

Send a POST request with the following JSON:

```json
{
  "tx_regularity_score": 80,
  "bill_payment_rate": 90,
  "income_stability": 75,
  "has_savings_behaviour": 1,
  "merchant_diversity": 5,
  "avg_monthly_inflow": 120,
  "account_age_months": 18
}
```

## Output

The API returns:

- `credit_score`
- `tier`
- `eligible`
- `loan_ceiling_naira`
- `feature_contributions`
- `explanation`

## Model plots

The project includes model evaluation and visualizations in `notebooks/03_plots.ipynb`. These plots demonstrate the feature relationships and scoring behavior used to validate the credit model.

## Next steps

- Build the frontend/UI for loan score delivery
- Add real OPay transaction features and live data integration
- Extend the model with additional behavioural signals
- Add caching, rate limiting, and deployment configuration

## Why this matters for OPay

- Uses OPay-native transaction and behavioural data
- Enables loan access without formal credit history or bank statements
- Supports OPay’s financial inclusion and credit expansion goals
- Provides a clear, Nigeria-specific value proposition for judges and stakeholders
