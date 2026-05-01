from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class MLScoreResult:
    probability_of_default: float
    score_300_900: int
    model_path: str


def profile_to_features(profile: Any) -> dict[str, Any]:
    """Convert an ApplicantProfile object into a flat feature dict.

    This is intentionally simple so you can train a scikit-learn pipeline using
    DictVectorizer/OneHotEncoder.
    """
    return {
        "age": getattr(profile, "age"),
        "employment_type": getattr(profile, "employment_type"),
        "years_employed": getattr(profile, "years_employed"),
        "annual_income": float(getattr(profile, "annual_income")),
        "monthly_debt": float(getattr(profile, "monthly_debt")),
        "loan_amount_requested": float(getattr(profile, "loan_amount_requested")),
        "loan_purpose": getattr(profile, "loan_purpose"),
        "credit_score": int(getattr(profile, "credit_score")),
        "missed_payments": int(getattr(profile, "missed_payments")),
        "savings": float(getattr(profile, "savings")),
        # Derived ratios (often predictive)
        "dti_ratio": float(getattr(profile, "dti_ratio")),
        "loan_to_income": float(getattr(profile, "loan_to_income")),
    }


def pd_to_score(pd: float) -> int:
    """Map probability of default (0..1) into a 300..900 score.

    This is a simple monotonic mapping for demo purposes.
    """
    pd_clamped = max(0.0, min(1.0, float(pd)))
    score = round(900 - 600 * pd_clamped)
    return int(max(300, min(900, score)))


def load_model(model_path: str) -> Any:
    try:
        import joblib  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: 'joblib'. Install ML deps (see README)."
        ) from e

    return joblib.load(model_path)


def score_with_model(model: Any, features: Mapping[str, Any]) -> float:
    """Return probability of default from a scikit-learn-like model/pipeline."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([dict(features)])
        return float(proba[0][1])

    if hasattr(model, "decision_function"):
        import math

        logit = float(model.decision_function([dict(features)])[0])
        return 1.0 / (1.0 + math.exp(-logit))

    raise RuntimeError(
        "Loaded ML model does not support predict_proba or decision_function."
    )


def maybe_score_profile(profile: Any, model_path: Optional[str] = None) -> Optional[MLScoreResult]:
    """Optionally compute an ML baseline score.

    Enable by setting `USE_ML_MODEL=1` and providing `ML_MODEL_PATH`.
    """
    use_ml = os.getenv("USE_ML_MODEL", "0").strip().lower() in {"1", "true", "yes", "y"}
    if not use_ml:
        return None

    resolved_path = (model_path or os.getenv("ML_MODEL_PATH", "")).strip()

    if not resolved_path:
        repo_root = Path(__file__).resolve().parent
        candidates = [
            repo_root / "models" / "applicant_credit_model.joblib",
            repo_root / "models" / "applicant_credit_model.pkl",
            repo_root / "models" / "credit_model.joblib",
            repo_root / "models" / "credit_model.pkl",
        ]
        for p in candidates:
            if p.exists():
                resolved_path = str(p)
                break

    if not resolved_path:
        raise RuntimeError(
            "USE_ML_MODEL is enabled but no ML model path was provided.\n\n"
            "Fix: train a model and set ML_MODEL_PATH. For example (PowerShell):\n"
            "  py generate_applicant_training_data.py --rows 5000 --out .\\data\\applicant_training.csv\n"
            "  py train_applicant_credit_model.py --data .\\data\\applicant_training.csv --out .\\models\\applicant_credit_model.joblib\n"
            "  $env:ML_MODEL_PATH = 'C:\\langchain_udemy\\models\\applicant_credit_model.joblib'"
        )

    model = load_model(resolved_path)
    features = profile_to_features(profile)
    pd = score_with_model(model, features)
    return MLScoreResult(probability_of_default=pd, score_300_900=pd_to_score(pd), model_path=resolved_path)
