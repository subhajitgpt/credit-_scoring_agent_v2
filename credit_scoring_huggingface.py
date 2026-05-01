# --- Ensure get_llm is importable at the top level ---
def get_llm() -> "BaseChatModel":
    """Public wrapper for _get_llm, for use in UI imports."""
    return _get_llm()

"""
Multi-Agent Credit Scoring System (Hugging Face Version)
=======================================================
This version uses Hugging Face Hub models for LLM agent reasoning.
"""

import os
import time
import textwrap
from dataclasses import dataclass
from typing import Literal, Optional

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.llms import HuggingFaceHub

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

MAX_TOKENS = 200

@dataclass
class ApplicantProfile:
    name: str
    age: int
    employment_type: str
    years_employed: int
    annual_income: float
    monthly_debt: float
    loan_amount_requested: float
    loan_purpose: str
    credit_score: int
    missed_payments: int
    savings: float

    @property
    def monthly_income(self) -> float:
        return self.annual_income / 12

    @property
    def dti_ratio(self) -> float:
        return (self.monthly_debt / self.monthly_income) * 100

    @property
    def loan_to_income(self) -> float:
        return self.loan_amount_requested / self.annual_income

    def to_summary(self) -> str:
        return textwrap.dedent(f"""
            Applicant      : {self.name}, Age {self.age}
            Employment     : {self.employment_type} ({self.years_employed} years)
            Annual Income  : ₹{self.annual_income:,.0f}
            Monthly Income : ₹{self.monthly_income:,.0f}
            Monthly Debt   : ₹{self.monthly_debt:,.0f}
            DTI Ratio      : {self.dti_ratio:.1f}%
            Loan Requested : ₹{self.loan_amount_requested:,.0f} (Purpose: {self.loan_purpose})
            Loan/Income    : {self.loan_to_income:.2f}x
            Credit Score   : {self.credit_score}/900
            Missed Payments: {self.missed_payments} (last 24 months)
            Savings/Assets : ₹{self.savings:,.0f}
        """).strip()

@dataclass
class AgentReport:
    agent_name: str
    sub_score: int
    analysis: str
    raw_response: str

@dataclass
class CreditDecision:
    final_score: int
    decision: str
    interest_rate: str
    max_loan_amount: float
    identity_score: int
    financial_score: int
    behavior_score: int
    risk_score: int
    summary: str
    conditions: Optional[str]

class SpecialistReport(BaseModel):
    sub_score: int = Field(..., ge=0, le=100)
    analysis: str = Field(..., min_length=1)

class OrchestratorDecision(BaseModel):
    final_score: int = Field(..., ge=300, le=900)
    decision: Literal["APPROVE", "CONDITIONAL", "REJECT"]
    interest_rate: str
    max_loan_amount: int = Field(..., ge=0)
    identity_score: int = Field(..., ge=0, le=100)
    financial_score: int = Field(..., ge=0, le=100)
    behavior_score: int = Field(..., ge=0, le=100)
    risk_score: int = Field(..., ge=0, le=100)
    summary: str
    conditions: Optional[str] = None

AGENT_SYSTEM_PROMPTS = {
    "identity": """You are the Identity & KYC Agent in a multi-agent credit scoring system for an Indian financial institution.\nYour role: assess identity trustworthiness, employment stability, and profile consistency.\nFocus on: employment type reliability, years employed vs age match, loan purpose legitimacy, red flags.\nReturn a concise analysis and a sub_score from 0-100 (higher = lower risk).\nDo NOT use protected attributes (religion, caste, race, gender, ethnicity) in the reasoning.""",
    "financial": """You are the Financial Health Agent in a multi-agent credit scoring system for an Indian financial institution.\nYour role: assess repayment capacity using income, existing debts, savings, and loan size.\nKey benchmarks: DTI < 40% is healthy, DTI 40-50% is borderline, DTI > 50% is concerning.\nLoan-to-income ratio > 5x is high risk. Savings should cover 3+ months of EMI.\nReturn a concise analysis and a sub_score from 0-100 (higher = lower risk).""",
    "behavioral": """You are the Behavioral Credit Agent in a multi-agent credit scoring system for an Indian financial institution.\nYour role: interpret credit bureau signals and payment behavior.\nKey benchmarks: CIBIL 750-900 = excellent, 650-749 = good, 550-649 = fair, 300-549 = poor.\nMissed payments in last 24 months are a major negative signal.\nReturn a concise analysis and a sub_score from 0-100 (higher = lower risk).""",
    "orchestrator": """You are the Risk Orchestrator — the final decision agent in a multi-agent credit scoring pipeline for an Indian financial institution.\nYou receive reports from specialist agents (including an ML baseline agent) and synthesize a final credit decision.\nReturn a final_score (300-900), decision, interest_rate, max_loan_amount, risk_score, summary, and optional conditions.\nBe consistent with specialist reports. Do NOT invent applicant details.""",
}

def _get_llm() -> BaseChatModel:
    """Return a Hugging Face Hub chat model."""
    repo_id = os.getenv("HF_REPO_ID", "google/gemma-7b")
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_token:
        raise RuntimeError("HUGGINGFACEHUB_API_TOKEN is not set. Add it to your environment or Streamlit secrets.")
    return HuggingFaceHub(
        repo_id=repo_id,
        huggingfacehub_api_token=api_token,
        model_kwargs={"temperature": float(os.getenv("HF_TEMPERATURE", "0.1")), "max_new_tokens": MAX_TOKENS}
    )

def _build_specialist_chain(llm: BaseChatModel, system_prompt: str) -> object:
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Evaluate this applicant:\n\n{profile_summary}"),
    ])
    return prompt | llm.with_structured_output(SpecialistReport)

def _run_specialist_agent(llm: BaseChatModel, profile: ApplicantProfile, *, key: str, name: str) -> AgentReport:
    chain = _build_specialist_chain(llm, AGENT_SYSTEM_PROMPTS[key])
    result: SpecialistReport = chain.invoke({"profile_summary": profile.to_summary()})
    return AgentReport(name, result.sub_score, result.analysis, result.model_dump_json())

def run_identity_agent(llm: BaseChatModel, profile: ApplicantProfile, *, verbose: bool = True) -> AgentReport:
    if verbose:
        print(f"\n  ▶ Agent 1 — Identity & KYC", end="", flush=True)
    report = _run_specialist_agent(llm, profile, key="identity", name="Identity & KYC")
    if verbose:
        print(f"  ✓ Score: {report.sub_score}/100")
    return report

def run_financial_agent(llm: BaseChatModel, profile: ApplicantProfile, *, verbose: bool = True) -> AgentReport:
    if verbose:
        print(f"  ▶ Agent 2 — Financial Health", end="", flush=True)
    report = _run_specialist_agent(llm, profile, key="financial", name="Financial Health")
    if verbose:
        print(f"  ✓ Score: {report.sub_score}/100")
    return report

def run_behavioral_agent(llm: BaseChatModel, profile: ApplicantProfile, *, verbose: bool = True) -> AgentReport:
    if verbose:
        print(f"  ▶ Agent 3 — Behavioral Credit", end="", flush=True)
    report = _run_specialist_agent(llm, profile, key="behavioral", name="Behavioral Credit")
    if verbose:
        print(f"  ✓ Score: {report.sub_score}/100")
    return report

def run_orchestrator(
    llm: BaseChatModel,
    profile: ApplicantProfile,
    ml_report: AgentReport,
    identity_report: AgentReport,
    financial_report: AgentReport,
    behavioral_report: AgentReport,
    *,
    verbose: bool = True,
) -> CreditDecision:
    if verbose:
        print(f"  ▶ Agent 4 — Risk Orchestrator", end="", flush=True)

    user_template = (
        "Applicant Data:\n{profile_summary}\n\n"
        "---\n"
        "AGENT 0 — Machine Learning Scoring (Score: {ml_score}/100):\n{ml_analysis}\n\n"
        "AGENT 1 — Identity & KYC (Score: {identity_score}/100):\n{identity_analysis}\n\n"
        "AGENT 2 — Financial Health (Score: {financial_score}/100):\n{financial_analysis}\n\n"
        "AGENT 3 — Behavioral Credit (Score: {behavior_score}/100):\n{behavior_analysis}\n\n"
        "Synthesize the above into a final credit decision."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", AGENT_SYSTEM_PROMPTS["orchestrator"]),
        ("user", user_template),
    ])

    chain = prompt | llm.with_structured_output(OrchestratorDecision)
    data: OrchestratorDecision = chain.invoke({
        "profile_summary": profile.to_summary(),
        "ml_score": ml_report.sub_score,
        "ml_analysis": ml_report.analysis,
        "identity_score": identity_report.sub_score,
        "identity_analysis": identity_report.analysis,
        "financial_score": financial_report.sub_score,
        "financial_analysis": financial_report.analysis,
        "behavior_score": behavioral_report.sub_score,
        "behavior_analysis": behavioral_report.analysis,
    })

    if verbose:
        print(f"  ✓ Decision: {data.decision}  |  Final Score: {data.final_score}")

    return CreditDecision(
        final_score=data.final_score,
        decision=data.decision,
        interest_rate=data.interest_rate,
        max_loan_amount=float(data.max_loan_amount),
        identity_score=data.identity_score,
        financial_score=data.financial_score,
        behavior_score=data.behavior_score,
        risk_score=data.risk_score,
        summary=data.summary,
        conditions=data.conditions,
    )

def _pd_to_sub_score(pd: float) -> int:
    pd_clamped = max(0.0, min(1.0, float(pd)))
    return int(round(100 * (1.0 - pd_clamped)))

def run_ml_scoring_agent(profile: ApplicantProfile, *, verbose: bool = True) -> AgentReport:
    if verbose:
        print(f"  ▶ Agent 0 — Machine Learning Scoring", end="", flush=True)
    try:
        from ml_credit_model import maybe_score_profile
        ml_result = maybe_score_profile(profile)
        if ml_result is None:
            analysis = (
                "ML baseline scoring is disabled. Set USE_ML_MODEL=1 and ML_MODEL_PATH to enable it."
            )
            if verbose:
                print(f"  ⚠ Score: 50/100")
            return AgentReport("Machine Learning Scoring", 50, analysis, raw_response="(disabled)")
        sub = _pd_to_sub_score(ml_result.probability_of_default)
        analysis = (
            f"Predicted PD={ml_result.probability_of_default:.3f} (lower is better). "
            f"Mapped baseline score≈{ml_result.score_300_900}/900. "
            f"Model: {ml_result.model_path}"
        )
        if verbose:
            print(f"  ✓ Score: {sub}/100")
        return AgentReport("Machine Learning Scoring", sub, analysis, raw_response=str(ml_result))
    except Exception as e:
        analysis = f"ML baseline unavailable ({e}). Proceeding without it."
        if verbose:
            print(f"  ⚠ Score: 50/100")
        return AgentReport("Machine Learning Scoring", 50, analysis, raw_response="(unavailable)")

def run_credit_scoring_pipeline(
    profile: ApplicantProfile,
    *,
    verbose: bool = True,
) -> CreditDecision:
    start = time.time()
    if load_dotenv is not None:
        load_dotenv()
    llm = _get_llm()
    provider = "huggingface"
    model_name = os.getenv("HF_REPO_ID", "google/gemma-7b")
    if verbose:
        print(f"\n{'─' * 60}")
        print(f"  Multi-Agent Credit Scoring Pipeline (Hugging Face)")
        print(f"  Applicant: {profile.name}  |  Model: {provider}:{model_name}")
        print(f"{'─' * 60}\n")
        print(f"  Running specialist agents...\n")
    ml_report = run_ml_scoring_agent(profile, verbose=verbose)
    identity_report = run_identity_agent(llm, profile, verbose=verbose)
    financial_report = run_financial_agent(llm, profile, verbose=verbose)
    behavioral_report = run_behavioral_agent(llm, profile, verbose=verbose)
    decision = run_orchestrator(
        llm,
        profile,
        ml_report,
        identity_report,
        financial_report,
        behavioral_report,
        verbose=verbose,
    )
    if verbose:
        elapsed_s = time.time() - start
        print(f"\n{'─' * 60}")
        print(f"  Specialist Agent Scores")
        print(f"  - Identity & KYC     : {identity_report.sub_score:>3}/100")
        print(f"  - Financial Health   : {financial_report.sub_score:>3}/100")
        print(f"  - Behavioral Credit  : {behavioral_report.sub_score:>3}/100")
        print(f"  - Risk (Orchestrator): {decision.risk_score:>3}/100")
        print(f"\n  Final Decision")
        print(f"  Decision     : {decision.decision}")
        print(f"  Final Score  : {decision.final_score}/900")
        print(f"  Interest Rate: {decision.interest_rate}")
        print(f"  Max Loan     : ₹{decision.max_loan_amount:,.0f}")
        if decision.conditions:
            print(f"  Conditions   : {decision.conditions}")
        print(f"\n  Completed in {elapsed_s:.1f}s")
        print(f"{'─' * 60}\n")
    return decision

def main() -> None:
    if load_dotenv is not None:
        load_dotenv()
    demo_profile = ApplicantProfile(
        name="Aarav Mehta",
        age=29,
        employment_type="Salaried",
        years_employed=6,
        annual_income=18_00_000,
        monthly_debt=22_000,
        loan_amount_requested=9_00_000,
        loan_purpose="Vehicle",
        credit_score=768,
        missed_payments=0,
        savings=3_50_000,
    )
    run_credit_scoring_pipeline(demo_profile)

if __name__ == "__main__":
    main()
