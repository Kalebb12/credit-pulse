import os
from dotenv import load_dotenv

load_dotenv()  # must be before os.environ is accessed

from google import genai
from google.genai import types
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from schema import ScoreRequest, ScoreResponse, FeatureContributions
from model import predict

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
app = FastAPI(title="CreditPulse API", version="1.0.0")

# Allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_explanation(score: int, data: dict, tier: str) -> str:
    prompt = f"""
You are a friendly financial advisor for OPay Nigeria.
A user just received a CreditPulse score of {score}/100 ({tier} tier).

Their profile:
- Bill payment rate: {data['bill_payment_rate']:.0f}%
- Transaction regularity: {data['tx_regularity_score']:.0f}%
- Income stability: {data['income_stability']:.0f}%
- Saves with OWealth: {'Yes' if data['has_savings_behaviour'] else 'No'}

In exactly 2 sentences:
1. Explain what drove their score in plain simple English.
2. Give one specific action they can take to improve it.

Keep it under 60 words. No jargon. Friendly tone.
"""
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            # config=types.GenerateContentConfig(
            #     max_output_tokens=100,
            #     temperature=0.4,
            # ),
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        return fallback_explanation(score)

def fallback_explanation(score: int) -> str:
    if score >= 60:
        return "Your consistent bill payments and regular transactions make you a reliable borrower. Keep saving with OWealth regularly to unlock higher credit limits."
    elif score >= 40:
        return "Your payment history shows some inconsistency which is affecting your score. Paying bills on time every month will move your score up significantly."
    else:
        return "Low transaction activity and irregular payments are holding your score back. Start with consistent small transactions on OPay for 60 days to begin building your profile."


@app.get("/")
def root():
    return {"status": "CreditPulse API is running"}


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest):
    data   = request.model_dump()
    result = predict(data)

    explanation = get_explanation(
        result["credit_score"], data, result["tier"]
    )

    return ScoreResponse(
        credit_score=result["credit_score"],
        tier=result["tier"],
        eligible=result["eligible"],
        loan_ceiling_naira=result["loan_ceiling_naira"],
        feature_contributions=FeatureContributions(
            **result["feature_contributions"]
        ),
        explanation=explanation,
    )