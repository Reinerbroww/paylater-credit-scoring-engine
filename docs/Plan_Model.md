## Model Comparison Experiments

The following models will be evaluated to compare different machine learning approaches, starting from a simple baseline model and progressing to advanced ensemble algorithms.

### 1. Logistic Regression

**Role:** Baseline Model

**Purpose:**
- Serves as the initial performance benchmark.
- Provides a simple and interpretable reference model.
- Helps measure how much performance improves when using more complex algorithms.

---

### 2. Random Forest

**Role:** Bagging Ensemble

**Purpose:**
- Provides a robust tree-based approach for tabular data.
- Uses multiple decision trees to improve generalization.
- Helps demonstrate the use of ensemble learning and techniques for reducing overfitting.

---

### 3. XGBoost

**Role:** Gradient Boosting

**Purpose:**
- Provides a powerful gradient boosting approach.
- Builds models sequentially to minimize prediction errors.
- Demonstrates an understanding of boosting and loss function optimization.

---

### 4. LightGBM

**Role:** Advanced Gradient Boosting

**Purpose:**
- Provides a fast and memory-efficient gradient boosting approach.
- Uses a leaf-wise tree growth strategy.
- Demonstrates consideration of computational efficiency and scalability.

---

## Experiment Flow

The models will be evaluated progressively:

**Logistic Regression → Random Forest → XGBoost → LightGBM**

Each model will be evaluated using the same evaluation metrics to ensure a fair comparison.