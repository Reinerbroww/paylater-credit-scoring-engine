"""Configuration for the PayLater credit scoring system.

Holds file paths, the engineered feature schema, decision-routing thresholds,
champion model hyperparameters, and categorical code sets shared by the
preprocessing and inference layers.
"""

MODEL_PATH = "model/lightgbm_paylater.pkl"
SCALER_PATH = "model/credit_data_scaler.pkl"

FEATURE_ORDER = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "person_home_ownership_OTHER",
    "person_home_ownership_OWN",
    "person_home_ownership_RENT",
    "loan_intent_EDUCATION",
    "loan_intent_HOMEIMPROVEMENT",
    "loan_intent_MEDICAL",
    "loan_intent_PERSONAL",
    "loan_intent_VENTURE",
    "loan_grade_B",
    "loan_grade_C",
    "loan_grade_D",
    "loan_grade_E",
    "loan_grade_F",
    "loan_grade_G",
    "cb_person_default_on_file_Y",
]

DECISION_LOW = 0.35
DECISION_HIGH = 0.70

MODEL_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.1,
    "scale_pos_weight": "auto (negatives / positives)",
    "random_state": 42,
}

MODEL_METRICS = {
    "roc_auc": "0.948",
    "precision": "88%",
    "recall": "78%",
    "accuracy": "93.0%",
}

HOME_CODES = ["RENT", "OWN", "MORTGAGE", "OTHER"]
INTENT_CODES = ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"]
GRADE_CODES = ["A", "B", "C", "D", "E", "F", "G"]
DEFAULT_CODES = ["N", "Y"]

APP_DEFAULTS = {
    "lang": "en",
    "in_income": 50000,
    "in_amnt": 5000,
    "in_age": 25,
    "in_rate": 11.5,
    "in_emp": 3.0,
    "in_hist": 4,
    "in_home": "RENT",
    "in_intent": "MEDICAL",
    "in_grade": "A",
    "in_default": "N",
    "th_low": DECISION_LOW,
    "th_high": DECISION_HIGH,
}

SAMPLE_PROFILES = {
    "Low risk": {
        "in_income": 92000, "in_amnt": 3000, "in_age": 36, "in_rate": 7.4,
        "in_emp": 9.0, "in_hist": 12, "in_home": "MORTGAGE", "in_intent": "EDUCATION",
        "in_grade": "A", "in_default": "N",
    },
    "Borderline": {
        "in_income": 45000, "in_amnt": 6000, "in_age": 26, "in_rate": 12.0,
        "in_emp": 3.0, "in_hist": 3, "in_home": "RENT", "in_intent": "MEDICAL",
        "in_grade": "D", "in_default": "N",
    },
    "High risk": {
        "in_income": 18000, "in_amnt": 19000, "in_age": 23, "in_rate": 18.5,
        "in_emp": 0.5, "in_hist": 1, "in_home": "RENT", "in_intent": "VENTURE",
        "in_grade": "E", "in_default": "Y",
    },
}
