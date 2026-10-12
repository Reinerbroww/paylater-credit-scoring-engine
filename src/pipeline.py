"""Inference pipeline.

Bundles data preparation and LightGBM inference behind a single entry point so
the UI layer stays decoupled from the model mechanics.
"""

import joblib

from .config import MODEL_PATH, SCALER_PATH
from .preprocess import build_feature_vector, load_scaler, scale_features


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def load_artifacts():
    return load_model(), load_scaler(SCALER_PATH)


def run_inference(model, scaler, features):
    scaled = scale_features(features, scaler)
    probability = float(model.predict_proba(scaled)[0, 1])
    contributions = model.booster_.predict(scaled, pred_contrib=True)[0][:-1]
    return probability, contributions


def classify(probability, low, high):
    if probability < low:
        return "approve"
    if probability <= high:
        return "review"
    return "reject"


def score_applicant(model, scaler, *, age, income, emp, amnt, rate, hist, home, intent, grade,
                    default, low, high):
    ratio = amnt / income if income > 0 else 0.0
    features = build_feature_vector(
        age=age,
        income=income,
        emp=emp,
        amnt=amnt,
        rate=rate,
        hist=hist,
        home=home,
        intent=intent,
        grade=grade,
        default=default,
        ratio=ratio,
    )
    probability, contributions = run_inference(model, scaler, features)
    return {
        "probability": probability,
        "contributions": contributions,
        "decision": classify(probability, low, high),
        "ratio": ratio,
    }
