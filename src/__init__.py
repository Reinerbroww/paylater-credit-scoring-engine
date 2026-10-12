"""Modular source package for the PayLater credit scoring system."""

from .config import DECISION_HIGH, DECISION_LOW, FEATURE_ORDER
from .pipeline import classify, load_artifacts, load_model, run_inference, score_applicant
from .preprocess import build_feature_vector, load_scaler, scale_features

__all__ = [
    "DECISION_LOW",
    "DECISION_HIGH",
    "FEATURE_ORDER",
    "build_feature_vector",
    "scale_features",
    "load_scaler",
    "load_model",
    "load_artifacts",
    "run_inference",
    "classify",
    "score_applicant",
]
