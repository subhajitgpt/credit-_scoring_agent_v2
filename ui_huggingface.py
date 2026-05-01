import os
os.environ["STREAMLIT_SERVER_PORT"] = "8505"
import streamlit as st
from credit_scoring_huggingface import (
    ApplicantProfile,
    get_llm as get_llm_hf,
    run_behavioral_agent,
    run_financial_agent,
    run_identity_agent,
    run_ml_scoring_agent,
    run_orchestrator,
)

st.set_page_config(page_title="Credit Scoring Agent Flow (Hugging Face)", layout="centered")
st.title("Credit Scoring Agent Flow (Hugging Face)")
st.caption("Model: Hugging Face Hub (see HF_REPO_ID)")

with st.form("applicant_form"):
    name = st.text_input("Name", value="Aarav Mehta")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=29, step=1)
        years_employed = st.number_input("Years employed", min_value=0, max_value=80, value=6, step=1)
        employment_type = st.selectbox(
            "Employment type",
            ["Salaried", "Self-Employed", "Business Owner", "Freelancer"],
            index=0,
        )
    with col2:
        credit_score = st.number_input("Credit score (300–900)", min_value=300, max_value=900, value=768, step=1)
        missed_payments = st.number_input("Missed payments (last 24 months)", min_value=0, max_value=100, value=0, step=1)
        loan_purpose = st.selectbox(
            "Loan purpose",
            ["Home", "Business", "Education", "Vehicle", "Personal"],
            index=3,
        )
    annual_income = st.number_input("Annual income (INR)", min_value=1.0, value=1_800_000.0, step=50_000.0)
    monthly_debt = st.number_input("Monthly debt / EMIs (INR)", min_value=0.0, value=22_000.0, step=1_000.0)
    savings = st.number_input("Savings / assets (INR)", min_value=0.0, value=350_000.0, step=10_000.0)
    loan_amount_requested = st.number_input("Loan amount requested (INR)", min_value=0.0, value=900_000.0, step=10_000.0)
    submitted = st.form_submit_button("Run flow")

def _render_agent(agent_number: str, title: str, score: int, analysis: str, raw: str) -> None:
    st.subheader(f"Agent {agent_number} — {title}")
    st.metric("Score", f"{score}/100")
    st.write(analysis)
    st.code(raw, language="json")

if submitted:
    try:
        profile = ApplicantProfile(
            name=name.strip(),
            age=int(age),
            employment_type=employment_type,
            years_employed=int(years_employed),
            annual_income=float(annual_income),
            monthly_debt=float(monthly_debt),
            loan_amount_requested=float(loan_amount_requested),
            loan_purpose=loan_purpose,
            credit_score=int(credit_score),
            missed_payments=int(missed_payments),
            savings=float(savings),
        )
        progress = st.progress(0, text="Stage 0/5 — Preparing profile")
        with st.status("Stage 0 — Applicant profile", expanded=True) as status:
            st.code(profile.to_summary())
            c1, c2, c3 = st.columns(3)
            c1.metric("Monthly income", f"₹{profile.monthly_income:,.0f}")
            c2.metric("DTI ratio", f"{profile.dti_ratio:.1f}%")
            c3.metric("Loan/Income", f"{profile.loan_to_income:.2f}x")
            status.update(label="Stage 0 — Applicant profile (ready)", state="complete")
        progress.progress(1 / 6, text="Stage 1/5 — Running ML baseline")
        llm = get_llm_hf()
        with st.status("Stage 1 — Agent 0 (ML baseline)", expanded=True) as status:
            ml_report = run_ml_scoring_agent(profile, verbose=False)
            _render_agent(
                "0",
                "Machine Learning Scoring",
                ml_report.sub_score,
                ml_report.analysis,
                ml_report.raw_response,
            )
            status.update(label="Stage 1 — Agent 0 (done)", state="complete")
        progress.progress(2 / 6, text="Stage 2/5 — Running Identity & KYC")
        with st.status("Stage 2 — Agent 1 (Identity & KYC)", expanded=True) as status:
            identity_report = run_identity_agent(llm, profile, verbose=False)
            _render_agent(
                "1",
                "Identity & KYC",
                identity_report.sub_score,
                identity_report.analysis,
                identity_report.raw_response,
            )
            status.update(label="Stage 2 — Agent 1 (done)", state="complete")
        progress.progress(3 / 6, text="Stage 3/5 — Running Financial Health")
        with st.status("Stage 3 — Agent 2 (Financial Health)", expanded=True) as status:
            financial_report = run_financial_agent(llm, profile, verbose=False)
            _render_agent(
                "2",
                "Financial Health",
                financial_report.sub_score,
                financial_report.analysis,
                financial_report.raw_response,
            )
            status.update(label="Stage 3 — Agent 2 (done)", state="complete")
        progress.progress(4 / 6, text="Stage 4/5 — Running Behavioral Credit")
        with st.status("Stage 4 — Agent 3 (Behavioral Credit)", expanded=True) as status:
            behavioral_report = run_behavioral_agent(llm, profile, verbose=False)
            _render_agent(
                "3",
                "Behavioral Credit",
                behavioral_report.sub_score,
                behavioral_report.analysis,
                behavioral_report.raw_response,
            )
            status.update(label="Stage 4 — Agent 3 (done)", state="complete")
        progress.progress(5 / 6, text="Stage 5/5 — Running Orchestrator")
        with st.status("Stage 5 — Agent 4 (Risk Orchestrator)", expanded=True) as status:
            decision = run_orchestrator(
                llm,
                profile,
                ml_report,
                identity_report,
                financial_report,
                behavioral_report,
                verbose=False,
            )
            st.subheader("Agent 4 — Risk Orchestrator")
            st.metric("Decision", decision.decision)
            c1, c2, c3 = st.columns(3)
            c1.metric("Final score", f"{decision.final_score}/900")
            c2.metric("Interest rate", decision.interest_rate)
            c3.metric("Max loan", f"₹{decision.max_loan_amount:,.0f}")
            st.write(decision.summary)
            if decision.conditions:
                st.info(f"Conditions: {decision.conditions}")
            status.update(label="Stage 5 — Agent 4 (done)", state="complete")
        progress.progress(1.0, text="Complete")
    except Exception as e:
        st.error(str(e))
        st.stop()
