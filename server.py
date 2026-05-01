from __future__ import annotations

import os
import logging
from dataclasses import asdict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from credit_scoring import ApplicantProfile, run_credit_scoring_pipeline


load_dotenv()

app = FastAPI(title="Credit Scoring Agent API", version="0.1.0")


@app.on_event("startup")
def _print_urls() -> None:
    host = os.getenv("HOST", "127.0.0.1")
    port = os.getenv("PORT", "8000")
    base = f"http://{host}:{port}"
    logger = logging.getLogger("uvicorn.error")
    logger.info("Credit Scoring Agent API is starting...")
    logger.info("Health: %s/health", base)
    logger.info("Docs  : %s/docs", base)
    logger.info("POST  : %s/credit-score", base)


class CreditScoreRequest(BaseModel):
    name: str = Field(..., min_length=1)
    age: int = Field(..., ge=18, le=100)
    employment_type: str
    years_employed: int = Field(..., ge=0, le=80)
    annual_income: float = Field(..., ge=0)
    monthly_debt: float = Field(..., ge=0)
    loan_amount_requested: float = Field(..., ge=0)
    loan_purpose: str
    credit_score: int = Field(..., ge=300, le=900)
    missed_payments: int = Field(..., ge=0, le=100)
    savings: float = Field(..., ge=0)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "llm_provider": os.getenv("LLM_PROVIDER", "ollama")}


@app.post("/credit-score")
def credit_score(payload: CreditScoreRequest) -> dict:
    try:
        profile = ApplicantProfile(**payload.model_dump())
        decision = run_credit_scoring_pipeline(profile, verbose=False)
        return asdict(decision)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
