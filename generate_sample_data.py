from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path


FEATURES = [
    "utilisation",
    "dpd_days",
    "cash_credit_ratio",
    "cash_debit_ratio",
    "inbound_cheque_bounce_count",
    "inbound_cheque_bounce_amt",
    "outbound_cheque_bounce_count",
    "outbound_cheque_bounce_amt",
    "total_amt_credit",
    "total_amt_debit",
    "no_of_banks",
]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sample_row(rng: random.Random) -> dict[str, float | int]:
    # Utilisation (0..1) skewed toward lower values
    utilisation = clamp(rng.betavariate(2.2, 4.8), 0.0, 1.0)

    # DPD days (0..120), heavy mass at 0
    if rng.random() < 0.75:
        dpd_days = 0
    else:
        dpd_days = int(clamp(rng.gammavariate(2.0, 12.0), 1.0, 120.0))

    # Cash ratios (0..1)
    cash_credit_ratio = clamp(rng.betavariate(2.5, 3.5), 0.0, 1.0)
    cash_debit_ratio = clamp(rng.betavariate(2.8, 3.2), 0.0, 1.0)

    # Amounts (lognormal-ish). Values are synthetic; treat as unitless.
    total_amt_credit = float(rng.lognormvariate(11.0, 0.8))
    # Debit tends to track credit but can exceed for risky profiles
    total_amt_debit = float(total_amt_credit * clamp(rng.lognormvariate(0.0, 0.35), 0.4, 2.2))

    # Banks
    no_of_banks = int(clamp(round(rng.gauss(3.5, 1.5)), 1.0, 10.0))

    # Cheque bounces: more likely for higher dpd/utilisation + lower credit inflow
    base_bounce_rate = 0.04 + 0.18 * utilisation + 0.005 * min(dpd_days, 60)
    inbound_bounce_count = int(rng.random() < base_bounce_rate) + int(rng.random() < base_bounce_rate / 2)
    outbound_bounce_count = int(rng.random() < base_bounce_rate * 1.1) + int(rng.random() < base_bounce_rate / 2)

    inbound_cheque_bounce_amt = float(inbound_bounce_count * rng.lognormvariate(9.5, 0.9))
    outbound_cheque_bounce_amt = float(outbound_bounce_count * rng.lognormvariate(9.7, 0.9))

    # Default probability (synthetic). Strongly driven by DPD, utilisation, bounces.
    z = (
        -3.0
        + 0.030 * dpd_days
        + 2.0 * utilisation
        + 0.35 * inbound_bounce_count
        + 0.40 * outbound_bounce_count
        + 0.9 * (cash_debit_ratio - 0.5)
        + 0.4 * (cash_credit_ratio - 0.5)
        + 0.15 * (no_of_banks - 3)
        + rng.gauss(0.0, 0.35)
    )
    pd = sigmoid(z)

    defaulted = 1 if rng.random() < pd else 0

    return {
        "utilisation": round(utilisation, 4),
        "dpd_days": dpd_days,
        "cash_credit_ratio": round(cash_credit_ratio, 4),
        "cash_debit_ratio": round(cash_debit_ratio, 4),
        "inbound_cheque_bounce_count": inbound_bounce_count,
        "inbound_cheque_bounce_amt": round(inbound_cheque_bounce_amt, 2),
        "outbound_cheque_bounce_count": outbound_bounce_count,
        "outbound_cheque_bounce_amt": round(outbound_cheque_bounce_amt, 2),
        "total_amt_credit": round(total_amt_credit, 2),
        "total_amt_debit": round(total_amt_debit, 2),
        "no_of_banks": no_of_banks,
        "defaulted": defaulted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic credit-scoring training data.")
    parser.add_argument("--out", type=Path, default=Path("data/credit_training_sample.csv"), help="Output CSV path")
    parser.add_argument("--rows", type=int, default=1000, help="Number of rows")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    if args.rows <= 0:
        raise SystemExit("--rows must be > 0")

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[*FEATURES, "defaulted"])
        writer.writeheader()
        defaults = 0
        for _ in range(args.rows):
            row = sample_row(rng)
            defaults += int(row["defaulted"])
            writer.writerow(row)

    print(f"Wrote {args.rows} rows to: {args.out}")
    print(f"Default rate: {defaults / args.rows:.2%}")


if __name__ == "__main__":
    main()
