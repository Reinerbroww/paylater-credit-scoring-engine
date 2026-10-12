# Real-Time Dynamic Credit Scoring System for E-Commerce PayLater

A production-style, real-time credit decision engine for e-commerce PayLater checkout.
It ingests an applicant's financial profile, engineers and normalizes the vector, runs a
champion LightGBM classifier, and routes the default probability through a three-band
decision gate — in under three seconds.

**Stack:** Python · LightGBM · scikit-learn · Streamlit · Joblib

---

## 1. Value Proposition

Traditional consumer credit relies on slow manual underwriting or delayed banking histories.
At the PayLater checkout this friction causes checkout abandonment and lost revenue.

This system replaces that bottleneck with a **real-time assessment engine** that evaluates an
applicant the moment they request financing. The model is deliberately weighted toward
**cash-flow capacity, depth of financial obligation, and risk-adjusted pricing** rather than
rigid demographic flags, which cuts false operational alarms roughly in half compared to a
weighted baseline and protects revenue while containing credit exposure.

**Operational targets**

| Objective | Target | Result |
|:--|:--|:--|
| Discriminatory power (ROC-AUC) | ≥ 0.94 | **0.9476** |
| Minority-class precision (default) | ≥ 85% | **88%** |
| End-to-end decision latency | < 3 s | **Sub-second** |

---

## 2. System Data Flow

The application layer is decoupled from the model core so the engine stays deterministic and
reusable (UI, API, or batch scoring):

```text
[ Streamlit UI Input Form ]
            |
            v
[ Feature Engineering  (ratio calc + one-hot encoding) ]   src/preprocess.py
            |
            v
[ Data Normalization   (fitted StandardScaler binary)  ]   src/preprocess.py
            |
            v
[ Champion Inference   (LightGBM leaf-wise classifier) ]   src/pipeline.py
            |
            v
[ Decision Routing Gate (threshold matrix alert output)]   src/pipeline.py
```

A single call to `score_applicant(...)` performs the entire chain and returns the probability,
per-feature contributions, decision band, and computed paycheck exposure.

---

## 3. Benchmarking Matrix

Three architectures were evaluated on the same stratified validation split to select the
production champion.

| Metric | Logistic Regression (weighted) | XGBoost | **LightGBM (champion)** |
|:--|:--:|:--:|:--:|
| Overall accuracy | 81.98% | 93.04% | **93.01%** |
| Default precision | 55.00% | 81.00% | **88.00%** |
| Default recall | 78.00% | 79.00% | **78.00%** |
| Default F1 | 0.65 | 0.80 | **0.83** |
| ROC-AUC | 0.8741 | 0.9472 | **0.9476** |

LightGBM delivers the highest ROC-AUC and F1, and a materially higher default precision than
XGBoost at equal recall — meaning fewer good applicants are wrongly blocked.

---

## 4. Gain Analysis

Feature importance by **gain** on the trained champion surfaces what actually drives risk:

| Rank | Feature | Share of total gain |
|:--:|:--|:--:|
| 1 | `loan_int_rate` (assigned interest rate) | 21.4% |
| 2 | `person_income` (annual income) | 19.4% |
| 3 | `loan_percent_income` (obligation / income) | 18.5% |
| 4 | `person_home_ownership_RENT` | 6.4% |
| 5 | `loan_amnt` (requested amount) | 5.0% |
| 6 | `person_age` | 4.4% |
| 7 | `person_emp_length` | 4.4% |

**Insight:** the top three features alone account for **~59% of total gain**. Risk here is
dominated by **pricing and cash-flow exposure** — the interest rate applied and the share of
income the requested credit line consumes — not by demographic signals. This is why the UI
computes and surfaces *paycheck exposure* (`loan_amnt / person_income`) as a first-class
underwriting metric.

**Decision routing** (probability of default):

| Band | Probability | Action |
|:--|:--|:--|
| Approve | `< 0.35` | Complete checkout instantly |
| Manual review | `0.35 – 0.70` | Halt checkout, route to underwriting |
| Reject | `> 0.70` | Block checkout to prevent credit loss |

Thresholds are configurable live from the dashboard sidebar, and every decision ships with a
per-feature contribution breakdown for explainability.

---

## 5. Future Roadmap

- **Phase 2 — Alternative Data Ingestion.** Fold e-commerce purchasing behaviour, historical
  payment punctuality, and live cart values into the risk matrix to score thin-file customers
  beyond banking history.
- **Phase 3 — Automated Threshold Tuning.** Drive approval cut-offs from a real-time business
  cost–benefit matrix, letting operators rebalance approval volume versus expected loss as
  macroeconomic conditions shift.

---

## Project Structure

```text
├── app.py                 # Streamlit UI (presentation only)
├── src/
│   ├── __init__.py        # public API surface
│   ├── config.py          # paths, feature schema, thresholds, hyperparams
│   ├── preprocess.py      # feature engineering + scaling
│   └── pipeline.py        # data prep + LightGBM inference + decision gate
├── model/
│   ├── lightgbm_paylater.pkl      # champion classifier
│   └── credit_data_scaler.pkl     # fitted StandardScaler
├── dataset/
│   └── credit_risk_dataset.csv
├── docs/                  # PRD, pipeline notes, wireframe, analysis
└── requirements.txt
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the dashboard
streamlit run app.py
```

The dashboard loads the champion model, accepts an applicant profile, and returns a decision
with a risk gauge, paycheck-exposure metrics, and an explainability chart. The UI supports
**English / Bahasa Indonesia**.

---

## Notes

- The entire decision path is reproducible from `src/pipeline.py::score_applicant`.
- This is a decision-support prototype: outputs are model estimates and do not constitute a
  final credit decision.
