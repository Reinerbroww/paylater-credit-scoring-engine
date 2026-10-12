## Week 1 — Data Preprocessing & Baseline Model

### 1. Data Preprocessing
- Check and handle **missing values** in relevant columns, such as:
  - Income
  - Employment length
  - Other features with missing values
- Perform **One-Hot Encoding** for categorical features, such as:
  - Home ownership type
  - Loan purpose
  - Other categorical variables

### 2. Train-Test Split
- Split the dataset into **training and testing sets**.
- Use **stratified splitting** (`stratify=y`) to maintain the target-class distribution in both subsets.
- Maintain an approximately **78:22 train-test ratio**.

### 3. Baseline Model
- Build a simple **Logistic Regression** model.
- Do not apply any class-imbalance handling at this stage.
- Use the result as the **baseline performance** for comparison with later experiments.

---

## Week 2 — Imbalance Handling & Advanced Algorithms

### 1. Class Weighting with XGBoost
- Train an **XGBoost** model using the `scale_pos_weight` parameter.
- Compare its performance against the baseline model.
- Record the changes in **F1-Score** and other relevant evaluation metrics.

### 2. SMOTE Experiment
- Apply **SMOTE (Synthetic Minority Over-sampling Technique)** to the training data.
- Train the selected model using the SMOTE-balanced dataset.
- Compare the results with the **class-weighting approach**.

> **Note:** SMOTE should only be applied to the training set to prevent data leakage.

### 3. Model Selection
- Compare the experimented models using:
  - **ROC-AUC**
  - **Recall**
  - **F1-Score**
- Select the model based on the combination of **ROC-AUC and Recall** performance.
- Carry the selected model forward to the **feature visualization stage in Week 3**.