from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path


EMPLOYMENT_TYPES = ["Salaried", "Self-Employed", "Business Owner", "Freelancer"]
LOAN_PURPOSES = ["Home", "Business", "Education", "Vehicle", "Personal"]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _sample_income(rng: random.Random) -> float:
    # INR annual income; lognormal-ish spread
    income = rng.lognormvariate(14.2, 0.55)  # ~ 1.5M typical
    return float(clamp(income, 180_000.0, 12_000_000.0))


def sample_row(rng: random.Random) -> dict[str, float | int | str]:
    age = int(clamp(round(rng.gauss(32, 8)), 21, 65))
    employment_type = rng.choices(
        EMPLOYMENT_TYPES,
        weights=[0.55, 0.20, 0.15, 0.10],
        k=1,
    )[0]

    # years employed depends on age and employment type
    max_years = max(0, age - 18)
    base_years = rng.randint(0, min(12, max_years))
    if employment_type == "Salaried":
        years_employed = int(clamp(round(base_years + rng.gauss(2.0, 2.0)), 0, max_years))
    elif employment_type == "Business Owner":
        years_employed = int(clamp(round(base_years + rng.gauss(1.0, 2.5)), 0, max_years))
    else:
        years_employed = int(clamp(round(base_years + rng.gauss(0.5, 2.5)), 0, max_years))

    annual_income = _sample_income(rng)
    monthly_income = annual_income / 12.0

    # debt as a fraction of income
    debt_ratio = clamp(rng.betavariate(2.2, 4.0), 0.0, 0.95)
    monthly_debt = float(round(monthly_income * debt_ratio, 2))

    loan_purpose = rng.choices(
        LOAN_PURPOSES,
        weights=[0.25, 0.20, 0.15, 0.20, 0.20],
        k=1,
    )[0]

    # loan size depends on purpose and income
    purpose_multiplier = {
        "Home": rng.uniform(2.0, 6.0),
        "Business": rng.uniform(1.5, 5.0),
        "Education": rng.uniform(0.8, 3.5),
        "Vehicle": rng.uniform(0.5, 2.0),
        "Personal": rng.uniform(0.3, 1.8),
    }[loan_purpose]
    loan_amount_requested = float(round(annual_income * purpose_multiplier, 2))

    # bureau-ish signals
    credit_score = int(clamp(round(rng.gauss(700, 85)), 300, 900))
    missed_payments = int(clamp(round(max(0.0, rng.gauss(0.6, 1.4))), 0, 12))

    # savings roughly related to income, with noise
    savings = float(round(clamp(rng.lognormvariate(12.5, 0.8), 0.0, 25_000_000.0), 2))

    # derived ratios
    dti_ratio = float(round((monthly_debt / monthly_income) * 100.0, 4))
    loan_to_income = float(round(loan_amount_requested / annual_income, 6))

    # synthetic default probability
    # Higher DTI, higher missed payments, lower bureau score, higher loan-to-income => higher PD
    employment_risk = {
        "Salaried": -0.25,
        "Self-Employed": 0.10,
        "Business Owner": 0.05,
        "Freelancer": 0.20,
    }[employment_type]

    # Map credit score into a risk term: 900->low risk, 300->high risk
    score_risk = (750 - credit_score) / 140.0

    # years employed stabilizes risk
    stability = -0.10 * min(years_employed, 10)

    z = (
        -2.0
        + 0.045 * (dti_ratio - 35.0)
        + 0.65 * max(0, missed_payments - 1)
        + 0.55 * max(0.0, loan_to_income - 2.0)
        + 0.55 * score_risk
        + employment_risk
        + stability
        + rng.gauss(0.0, 0.35)
    )

    pd = sigmoid(z)
    defaulted = 1 if rng.random() < pd else 0

    return {
        "age": age,
        "employment_type": employment_type,
        "years_employed": years_employed,
        "annual_income": round(annual_income, 2),
        "monthly_debt": round(monthly_debt, 2),
        "loan_amount_requested": round(loan_amount_requested, 2),
        "loan_purpose": loan_purpose,
        "credit_score": credit_score,
        "missed_payments": missed_payments,
        "savings": round(savings, 2),
        "dti_ratio": round(dti_ratio, 4),
        "loan_to_income": round(loan_to_income, 6),
        "defaulted": defaulted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic ApplicantProfile-style training data for the ML baseline agent."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/applicant_training.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--rows", type=int, default=5000, help="Number of rows")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    if args.rows <= 0:
        raise SystemExit("--rows must be > 0")

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

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

    defaults = 0
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for _ in range(args.rows):
            row = sample_row(rng)
            defaults += int(row["defaulted"])  # type: ignore[arg-type]
            writer.writerow(row)

    print(f"Wrote {args.rows} rows to: {args.out}")
    print(f"Default rate: {defaults / args.rows:.2%}")


if __name__ == "__main__":
    main()
