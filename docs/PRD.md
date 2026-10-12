# Product Requirements Document (PRD)

## 1. Document Overview

- **Project Title:** Real-Time Dynamic Credit Scoring System for E-Commerce PayLater
- **Target Stack:** Python, LightGBM, Scikit-learn, Streamlit, Joblib
- **Document Version:** 1.0.0
- **Date:** October 12, 2026

---

## 2. Executive Summary & Value Proposition

In modern digital ecosystems, consumer credit systems (PayLater) require rapid, friction-free verification mechanisms to approve transactions at checkout. Traditional credit scoring relies heavily on manual underwriting or slow banking histories, leading to high transaction abandonment rates.

This product addresses this bottleneck by deploying a **highly optimized, real-time assessment dashboard** powered by a champion machine learning model. By prioritizing cash flow, depth of financial obligation, and dynamic risk-adjusted pricing over rigid structures, the engine cuts false operational alarms nearly in half compared to baseline algorithms, protecting business revenue while containing credit exposure.

---

## 3. Core Objectives & Metrics

- **Risk Control:** Achieve robust discriminatory power between reliable borrowers and potential default risks, targeting an operational Area Under the Curve (ROC-AUC) score of **≥ 0.94**.
- **Frictionless Checkout:** Minimize false rejections for transaction financing by optimizing classification thresholds to maintain a minority class **precision score of ≥ 85%**.
- **System Latency:** Process incoming applicant attributes, calculate complex financial ratios, scale vectors, and render credit decisions in under **3 seconds**.

---

## 4. System Architecture & Data Pipeline

The application layer follows a decoupled processing workflow to guarantee rapid, deterministic model execution:

```text
[User Interface (Streamlit UI Input Form)]
                  |
                  v
[Feature Engineering Module (Ratio Calculations & Dummy Coding)]
                  |
                  v
[Data Normalization Pipeline (Fitted sklearn StandardScaler Instance File)]
                  |
                  v
[Champion Inference Core (Leaf-wise LightGBM Classifier PKL Object)]
                  |
                  v
[Decision Routing Gate (Threshold Matrix Alert Output UI Component)]
```

---

## 5. Functional Requirements (Product Features)

### 5.1 Interactive Financial Profile Input Interface

- **Feature Requirement:** The system must provide custom validation fields to ingest customer vectors.

- **Input Parameters:**
  - **Numerical Fields:** Annual Income (`person_income`), Requested Loan Amount (`loan_amnt`), Assigned Interest Rate (`loan_int_rate`), Customer Age (`person_age`), and Employment History (`person_emp_length`).
  - **Categorical Fields:** Home Ownership Structure Dropdown (`RENT`, `OWN`, `MORTGAGE`, `OTHER`).
  - **Dynamic Calculations:** Automated client-side calculations computing the percentage of an applicant's paycheck the requested credit line will consume (`loan_percent_income`).

### 5.2 Preprocessing Integrity Framework

- **Feature Requirement:** The application backend must maintain state consistency by enforcing identical feature engineering transformations used during the model optimization phase.

- **Technical Constraints:**
  - Categorical input categories must dynamically translate to their precise one-hot encoded counterparts.
  - Inbound arrays must pass through a pre-fitted scalar binary (`credit_data_scaler.pkl`) before submission to the model object to prevent feature magnitude bias.

### 5.3 Automated Decision Gates (Inference Pipeline)

- **Feature Requirement:** The interface must route the predictive probabilities generated from the serialized champion LightGBM object (`lightgbm_paylater_champion.pkl`) into defined decision handling categories:

  - **Probability Score < 0.35:** Trigger full operational approval (`APPLICATION APPROVED`).
  - **Probability Score 0.35 to 0.70:** Halt instant checkout and route vector traits to underwriting logs (`MANUAL REVIEW REQUIRED`).
  - **Probability Score > 0.70:** Block checkout transaction to prevent credit losses (`APPLICATION REJECTED`).

---

## 6. Model Performance & Evaluation Baseline

The structural foundation of this system relies on a rigorous tournament assessment phase against competitive architectures, with the production settings benchmarked on a validation set.

| Performance Metric | Baseline Logistic Regression (Weighted) | Secondary XGBoost Classifier | Production Champion (LightGBM) |
|:---|:---:|:---:|:---:|
| **Overall Classification Accuracy** | 81.98% | 93.04% | **93.01%** |
| **Class 1 (Default) Precision** | 55.00% | 81.00% | **88.00%** |
| **Class 1 (Default) Recall** | 78.00% | 79.00% | **78.00%** |
| **Class 1 (Default) F1-Score** | 0.65 | 0.80 | **0.83** |
| **Discriminatory Power (ROC-AUC)** | 0.8741 | 0.9472 | **0.9476** |

---

## 7. Future Roadmap & Iterations

- **Phase 2 (Alternative Data Ingestion):** Incorporate historical e-commerce purchasing habits, past payment punctuality metrics, and platform cart values directly into the risk modeling matrix.

- **Phase 3 (Threshold Tuning):** Implement real-time business cost-benefit matrices to allow operational headers to adjust approval cutoff boundaries dynamically based on macroeconomic conditions.
