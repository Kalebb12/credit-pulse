from pydantic import BaseModel, Field

class ScoreRequest(BaseModel):
    tx_regularity_score:   float = Field(..., ge=0,  le=100)
    bill_payment_rate:     float = Field(..., ge=0,  le=100)
    income_stability:      float = Field(..., ge=0,  le=100)
    has_savings_behaviour: int   = Field(..., ge=0,  le=1)
    merchant_diversity:    int   = Field(..., ge=0,  le=15)
    avg_monthly_inflow:    float = Field(..., ge=5,  le=500)
    account_age_months:    int   = Field(..., ge=1,  le=48)

class FeatureContributions(BaseModel):
    tx_regularity_score:   float
    bill_payment_rate:     float
    income_stability:      float
    has_savings_behaviour: float
    merchant_diversity:    float
    avg_monthly_inflow:    float
    account_age_months:    float

class ScoreResponse(BaseModel):
    credit_score:          int
    tier:                  str
    eligible:              bool
    loan_ceiling_naira:    int
    feature_contributions: FeatureContributions
    explanation:           str