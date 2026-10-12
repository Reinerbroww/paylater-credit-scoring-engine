# Week One Analysis

## 1. Data Understanding

The dataset contains **32,581 rows and 12 columns**. After inspecting the dataset, I selected **`loan_status`** as the target variable (label) for the classification task.

The `loan_status` variable contains two classes:

* **0 = Non-default**
* **1 = Default**

The class distribution is:

* **25,473 records (78%)** → Non-default
* **7,108 records (22%)** → Default

This shows that the dataset is **imbalanced**, because the number of non-default records is considerably higher than the number of default records. Therefore, class imbalance needs to be considered in the next stages of the analysis to prevent the model from being overly influenced by the majority class.

The dataset contains both **categorical and numerical features**. Categorical features were transformed from string values into numerical representations using **One-Hot Encoding**, allowing them to be processed by machine learning algorithms.

During the data exploration process, I also found several feature values that appeared unrealistic or inconsistent with real-world conditions. For example, some values related to **age and employment length** required adjustment so that the data would better represent realistic conditions.

After the initial feature processing, the dataset was separated into:

* **X** → input features used by the model
* **y** → target variable (`loan_status`)

I also created a bar chart to visualize the distribution of the target variable and confirm the class imbalance identified during the initial analysis.

---

## 2. Data Preparation

The next step was preparing the dataset for modeling.

First, I checked the dataset for missing values. The only feature containing missing values was **`loan_int_rate`**. To handle these missing values, I replaced them with the **mean value** of the feature. This approach was selected because the missing values were numerical and using the mean allowed the dataset to remain complete without removing additional observations.

I also checked for duplicate records and found **157 duplicate rows**. These duplicate records were removed because keeping them could cause certain observations to have a greater influence on the model than intended.

After removing the duplicate records, the dataset contained:

* **31,522 rows**
* **12 columns**

The cleaned dataset was then divided into **training and testing data** using a **78:22 split**. The training data was used to build the model, while the testing data was used to evaluate how well the model performed on unseen data.

Finally, feature scaling was applied to the dataset so that numerical features could be placed on a comparable scale before being used by the machine learning model.

---

## 3. Modeling

For Week One, I used **Linear Regression** as an initial baseline model.

The model was trained using the prepared training dataset and then evaluated using the testing dataset. The model achieved an accuracy of approximately:

**86.96%**

This result provides a baseline for the following experiments. However, because the target variable is a classification problem with two classes, further experiments with classification algorithms will be necessary to determine whether the model can identify both **default and non-default** cases effectively, especially considering the class imbalance identified during data understanding.

---

## 4. Week One Conclusion

Overall, Week One focused on understanding and preparing the loan dataset before applying more advanced machine learning methods.

The main findings were:

1. The dataset initially contained **32,581 records and 12 features**.
2. The target variable was **`loan_status`**, consisting of non-default (0) and default (1).
3. The target distribution was **78% non-default and 22% default**, indicating class imbalance.
4. Categorical features were converted into numerical representations using **One-Hot Encoding**.
5. Unrealistic values in features such as age and employment length were identified and adjusted.
6. Missing values in **`loan_int_rate`** were handled using the mean.
7. **157 duplicate records** were removed, leaving **31,522 records**.
8. The dataset was split into **training and testing sets using a 78:22 ratio**.
9. Feature scaling was applied before modeling.
10. Linear Regression achieved an initial accuracy of **86.96%**.

The results from Week One will serve as a **baseline** for the next experiments, where different machine learning algorithms and data-balancing techniques can be tested and compared.



# Week Two Analysis
