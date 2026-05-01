from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def _to_number_or_str(value: str) -> float | str:
    v = value.strip()
    if v == "":
        return 0.0
    try:
        return float(v)
    except Exception:
        return v


def read_training_csv(path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    """Reads ApplicantProfile-style training CSV with a binary `defaulted` column."""

    X: list[dict[str, Any]] = []
    y: list[int] = []

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header.")
        if "defaulted" not in reader.fieldnames:
            raise ValueError("CSV must include a 'defaulted' column (0/1).")

        feature_columns = [c for c in reader.fieldnames if c != "defaulted"]
        if not feature_columns:
            raise ValueError("CSV has no feature columns.")

        for row in reader:
            target = int(float(row["defaulted"].strip()))
            features: dict[str, Any] = {}
            for col in feature_columns:
                features[col] = _to_number_or_str(row[col])
            X.append(features)
            y.append(target)

    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an ApplicantProfile-style credit default model and save it as joblib."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/applicant_training.csv"),
        help="Path to training CSV containing a 'defaulted' column",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/applicant_credit_model.joblib"),
        help="Output joblib path",
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

    X, y = read_training_csv(args.data)
    if len(X) < 200:
        raise RuntimeError("Need more training rows (try 2000+ for a meaningful demo).")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        steps=[
            ("vec", DictVectorizer(sparse=True)),
            ("clf", LogisticRegression(max_iter=300, n_jobs=None)),
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
