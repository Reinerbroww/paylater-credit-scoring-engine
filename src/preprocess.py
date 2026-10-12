"""Preprocessing layer.

Rebuilds the exact feature engineering and scaling transformations used during
model training, so inbound applicant vectors match the training distribution.
"""

import joblib
import pandas as pd

from .config import FEATURE_ORDER, SCALER_PATH


def load_scaler(path=SCALER_PATH):
    return joblib.load(path)


def build_feature_vector(age, income, emp, amnt, rate, hist, home, intent, grade, default, ratio):
    row = {key: 0.0 for key in FEATURE_ORDER}
    row["person_age"] = float(age)
    row["person_income"] = float(income)
    row["person_emp_length"] = float(emp)
    row["loan_amnt"] = float(amnt)
    row["loan_int_rate"] = float(rate)
    row["loan_percent_income"] = float(ratio)
    row["cb_person_cred_hist_length"] = float(hist)

    home_key = f"person_home_ownership_{home}"
    if home_key in row:
        row[home_key] = 1.0

    intent_key = f"loan_intent_{intent}"
    if intent_key in row:
        row[intent_key] = 1.0

    grade_key = f"loan_grade_{grade}"
    if grade_key in row:
        row[grade_key] = 1.0

    if default == "Y":
        row["cb_person_default_on_file_Y"] = 1.0

    return pd.DataFrame([[row[k] for k in FEATURE_ORDER]], columns=FEATURE_ORDER)


def scale_features(features, scaler):
    return pd.DataFrame(scaler.transform(features), columns=FEATURE_ORDER)
