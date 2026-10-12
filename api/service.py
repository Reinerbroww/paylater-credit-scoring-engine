"""Scoring service.

Loads the champion model and scaler once and exposes a single entry point that
turns an ApplicantRequest into a scored decision, keeping the API layer thin.
"""

from functools import lru_cache

from src.config import (
    DECISION_HIGH,
    DECISION_LOW,
    FEATURE_ORDER,
    MODEL_METRICS,
    MODEL_PARAMS,
)
from src.pipeline import load_artifacts, score_applicant

MODEL_NAME = "LightGBM PayLater Champion"
MODEL_ALGORITHM = "LightGBM (leaf-wise gradient boosting)"


class ScoringService:
    def __init__(self):
        self.model, self.scaler = load_artifacts()

    @property
    def features(self):
        return list(FEATURE_ORDER)

    def info(self):
        return {
            "model": MODEL_NAME,
            "algorithm": MODEL_ALGORITHM,
            "feature_count": len(FEATURE_ORDER),
            "features": list(FEATURE_ORDER),
            "metrics": dict(MODEL_METRICS),
            "params": dict(MODEL_PARAMS),
            "default_thresholds": {"low": DECISION_LOW, "high": DECISION_HIGH},
            "health": "ok",
        }

    def score(self, request):
        low = min(request.threshold_low, request.threshold_high)
        high = max(request.threshold_low, request.threshold_high)
        result = score_applicant(
            self.model,
            self.scaler,
            age=request.person_age,
            income=request.person_income,
            emp=request.person_emp_length,
            amnt=request.loan_amnt,
            rate=request.loan_int_rate,
            hist=request.cb_person_cred_hist_length,
            home=request.person_home_ownership,
            intent=request.loan_intent,
            grade=request.loan_grade,
            default=request.cb_person_default_on_file,
            low=low,
            high=high,
        )
        contributions = [
            {"feature": name, "value": float(value)}
            for name, value in zip(FEATURE_ORDER, result["contributions"])
        ]
        return {
            "probability": result["probability"],
            "decision": result["decision"],
            "loan_percent_income": result["ratio"],
            "contributions": contributions,
            "model": MODEL_NAME,
            "threshold_low": low,
            "threshold_high": high,
        }


@lru_cache(maxsize=1)
def get_service():
    return ScoringService()
