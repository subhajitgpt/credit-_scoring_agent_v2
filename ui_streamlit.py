from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


from credit_scoring import (
    ApplicantProfile,
    get_llm,
    run_behavioral_agent,
    run_financial_agent,
    run_identity_agent,
    run_ml_scoring_agent,
    run_orchestrator,
)




def _load_env() -> None:
    if load_dotenv is not None:
        # Streamlit doesn't automatically load .env
        load_dotenv(override=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_ml_model_path() -> Path:
    return _repo_root() / "models" / "applicant_credit_model.pkl"


def _ml_enabled() -> bool:
    return os.getenv("USE_ML_MODEL", "0").strip().lower() in {"1", "true", "yes", "y"}


def _resolve_ml_model_path() -> str:
    raw = os.getenv("ML_MODEL_PATH", "").strip()
    if raw:
        return raw
    p = _default_ml_model_path()
    return str(p) if p.exists() else ""


import streamlit as st
@st.cache_resource(show_spinner=False)
def _bootstrap_demo_ml_model(*, rows: int = 3000) -> Path:
    """Create a small demo ML model matching the Streamlit form inputs."""

    data_path = _repo_root() / "data" / "applicant_training.csv"
    model_path = _default_ml_model_path()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    # Heavy imports inside function
    import generate_applicant_training_data as gen
    import csv
    import joblib  # type: ignore
    from sklearn.feature_extraction import DictVectorizer  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore
    from sklearn.pipeline import Pipeline  # type: ignore
    import train_applicant_credit_model as trainer

    rng = __import__("random").Random(42)
    fieldnames = [
        "age",
        "employment_type",
        "years_employed",
        "annual_income",
        "monthly_debt",
        "loan_amount_requested",
        "loan_purpose",
        "credit_score",
        "missed_payments",
        "savings",
        "dti_ratio",
        "loan_to_income",
        "defaulted",
    ]

    with data_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for _ in range(rows):
            writer.writerow(gen.sample_row(rng))

    X, y = trainer.read_training_csv(data_path)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = Pipeline(
        steps=[
            ("vec", DictVectorizer(sparse=True)),
            ("clf", LogisticRegression(max_iter=300, n_jobs=None)),
        ]
    )
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    return model_path


_load_env()


def _env_label() -> str:
    provider = os.getenv("LLM_PROVIDER", "ollama")
    if provider.strip().lower() == "ollama":
        model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    else:
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
    return f"{provider}:{model}"


st.set_page_config(page_title="Credit Scoring Agent Flow", layout="centered")

st.title("Credit Scoring Agent Flow")
st.caption(f"Model: {_env_label()}")

with st.sidebar:
    st.subheader("ML baseline")
    enabled = _ml_enabled()
    resolved = _resolve_ml_model_path()
    st.write(f"USE_ML_MODEL: `{os.getenv('USE_ML_MODEL', '0')}`")
    st.write(f"ML_MODEL_PATH: `{os.getenv('ML_MODEL_PATH', '')}`")
    if st.button("Generate demo ML model (.pkl)"):
        with st.spinner("Generating training data + training model..."):
            path = _bootstrap_demo_ml_model(rows=3000)
        os.environ["USE_ML_MODEL"] = "1"
        os.environ["ML_MODEL_PATH"] = str(path)
        st.success(f"Created: {path}")
        st.rerun()

    if enabled and resolved:
        st.success(f"ML baseline active: {resolved}")
    elif enabled and not resolved:
        st.warning(
            "ML baseline is enabled but no model file was found. Click 'Generate demo ML model (.pkl)'."
        )
    else:
        st.info("ML baseline currently disabled. Click 'Generate demo ML model (.pkl)' to enable.")

with st.form("applicant_form"):
    name = st.text_input("Name", value="Aarav Mehta")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=29, step=1)
        years_employed = st.number_input(
            "Years employed", min_value=0, max_value=80, value=6, step=1
        )
        employment_type = st.selectbox(
            "Employment type",
            ["Salaried", "Self-Employed", "Business Owner", "Freelancer"],
            index=0,
        )

    with col2:
        credit_score = st.number_input(
            "Credit score (300–900)", min_value=300, max_value=900, value=768, step=1
        )
        missed_payments = st.number_input(
            "Missed payments (last 24 months)", min_value=0, max_value=100, value=0, step=1
        )
        loan_purpose = st.selectbox(
            "Loan purpose",
            ["Home", "Business", "Education", "Vehicle", "Personal"],
            index=3,
        )

    annual_income = st.number_input(
        "Annual income (INR)", min_value=1.0, value=1_800_000.0, step=50_000.0
    )
    monthly_debt = st.number_input(
        "Monthly debt / EMIs (INR)", min_value=0.0, value=22_000.0, step=1_000.0
    )
    savings = st.number_input(
        "Savings / assets (INR)", min_value=0.0, value=350_000.0, step=10_000.0
    )
    loan_amount_requested = st.number_input(
        "Loan amount requested (INR)", min_value=0.0, value=900_000.0, step=10_000.0
    )

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

        llm = get_llm()

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
        message = str(e)
        lower = message.lower()

        if (
            "requires more system memory" in lower
            or "llama runner process has terminated" in lower
            or "status code: 500" in lower and "ollama" in lower
        ):
            st.error("Ollama model runner crashed (likely out of RAM).")
            st.code(message)
            st.markdown(
                "Try a smaller model, then re-run the flow:\n\n"
                "- `ollama pull llama3.2:3b`\n"
                "- PowerShell: `$env:OLLAMA_MODEL = 'llama3.2:3b'`\n"
                "- Restart Streamlit (or rerun)."
            )
        else:
            st.error(message)
        st.stop()
