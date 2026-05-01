from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


DEFAULT_FEATURES: list[str] = [
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


def _to_int(value: str) -> int:
    return int(float(value.strip()))


def _to_float(value: str) -> float:
    return float(value.strip())


def read_training_csv(path: Path, feature_columns: list[str]) -> tuple[list[dict[str, Any]], list[int]]:
    """Reads a CSV with numeric features + a binary `defaulted` column."""

    X: list[dict[str, Any]] = []
    y: list[int] = []

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header.")
        if "defaulted" not in reader.fieldnames:
            raise ValueError("CSV must include a 'defaulted' column (0/1).")

        missing = [c for c in feature_columns if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV is missing required feature columns: {missing}")

        for row in reader:
            target = _to_int(row["defaulted"])

            features: dict[str, Any] = {}
            for col in feature_columns:
                features[col] = _to_float(row[col])

            X.append(features)
            y.append(int(target))

    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple credit default model and save it as joblib.")
    parser.add_argument("--data", type=Path, required=True, help="Path to training CSV containing a 'defaulted' column")
    parser.add_argument("--out", type=Path, default=Path("models/credit_model.joblib"), help="Output joblib path")
    parser.add_argument(
        "--features",
        type=str,
        default=",".join(DEFAULT_FEATURES),
        help="Comma-separated feature column names to use for training",
    )
    args = parser.parse_args()

    try:
        import joblib  # type: ignore
        from sklearn.feature_extraction import DictVectorizer  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.metrics import roc_auc_score  # type: ignore
        from sklearn.model_selection import train_test_split  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing ML dependencies. Install: py -m pip install scikit-learn joblib"
        ) from e

    feature_columns = [c.strip() for c in str(args.features).split(",") if c.strip()]
    if not feature_columns:
        raise RuntimeError("No feature columns provided. Use --features.")

    X, y = read_training_csv(args.data, feature_columns)
    if len(X) < 50:
        raise RuntimeError("Need more training rows (try 500+ for a meaningful demo).")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = Pipeline(
        steps=[
            ("vec", DictVectorizer(sparse=True)),
            ("clf", LogisticRegression(max_iter=200, n_jobs=None)),
        ]
    )

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)

    print(f"Saved model: {args.out}")
    print(f"Test ROC-AUC : {auc:.3f}")


if __name__ == "__main__":
    main()
