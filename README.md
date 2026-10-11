# 📊 Telecom Customer Churn Prediction (Logistic Regression)

An end-to-end Machine Learning pipeline and business decision framework predicting customer churn using Cost-Sensitive Logistic Regression in Python.

---

## 🎯 Objective
The primary objective of this project is to analyze customer demographics, account information, and subscription services to predict customer attrition (churn) for a telecommunications provider. By accurately identifying customers at high risk of defecting to competitors, management can deploy targeted retention strategies, reduce subscriber acquisition costs, and protect recurring revenue.

---

## 🛠️ Technologies Used
- **Programming Language:** Python 3.x
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Machine Learning & Preprocessing:** Scikit-Learn (`LogisticRegression`, `Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `SimpleImputer`)
- **Evaluation Metrics:** Accuracy, Precision, Recall (Sensitivity), F1-Score, ROC-AUC, Confusion Matrix
- **Development Environment:** VS Code, Jupyter Notebook (`customer_churn_analysis.ipynb`)

---

## 📊 Dataset Architecture & Features
The model trains on **7,043 customer records** across 20 distinct attributes from `Logistic Regression.csv`:
- **Customer Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Account & Financial Terms:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- **Service Portfolio:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Target Variable:** `Churn` ('Yes' / 'No' → 1 / 0)

---

## ⚙️ Methodology & Pipeline Architecture

1. **Data Hygiene & Preprocessing:**
   - Coerced `TotalCharges` to numeric and resolved missing entries via median imputation.
   - Identified class imbalance: ~2.76 retained customers for every 1 churning customer (26.54% churn rate).

2. **Modular Scikit-Learn Pipeline (`ColumnTransformer`):**
   - **Numerical Pipeline:** Median imputation + standard scaling (`StandardScaler`) for `tenure`, `MonthlyCharges`, and `TotalCharges`.
   - **Categorical Pipeline:** Most frequent imputation + one-hot encoding (`OneHotEncoder(drop='first')`) across 15 categorical features.

3. **Cost-Sensitive Logistic Regression Training:**
   - Partitioned dataset into **80% training** (5,634 samples) and **20% testing** (1,409 samples) using stratified sampling.
   - Applied **L2 regularization** ($C=0.5$) with balanced cost weights (`class_weight={0: 1.0, 1: 2.2}`) to optimize recall for the minority churn class.

4. **Probability Segmentation & Unseen Customer Profiling:**
   - Evaluated probability thresholds at 0.50.
   - Tested real-world synthetic profiles (Month-to-month fiber optic vs. 2-year DSL contract) to trigger proactive retention offers.

---

## 📈 Key Results & Performance Evaluation

| Evaluation Metric | Score | Business Interpretation |
| :--- | :--- | :--- |
| **Overall Accuracy** | **~75% - 78%** | Robust generalization on holdout test partition |
| **Churn Recall (Sensitivity)** | **~75%+** | Successfully captures the vast majority of defecting customers |
| **F1-Score** | **~0.63** | Harmonic balance between Precision and Recall |
| **ROC-AUC Score** | **~0.84** | High discriminatory capability between loyal and churning accounts |

### Error Cost Trade-Off Analysis:
- **Type I Error (False Alarm):** Flagging a loyal customer results only in a small promotional discount cost.
- **Type II Error (Missed Defection):** Missing an at-risk customer results in complete loss of recurring lifetime subscription revenue and high re-acquisition costs.
- **Conclusion:** Prioritizing high **Churn Recall** significantly outperforms standard accuracy maximization for corporate profitability.

---

## 💡 Strategic Recommendations

1. **Target Early Tenure:** Focus onboarding check-ins and customer satisfaction audits within the **first 12 months** of tenure.
2. **Contract Migration:** Incentivize month-to-month customers to switch to 1-year or 2-year contracts with introductory discounts.
3. **Service Bundling:** Bundle complimentary **TechSupport** and **OnlineSecurity** with Fiber Optic subscriptions to mitigate elevated churn rates.

---

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/mahimavarshini/customer-churn-logistic-regression.git
   cd customer-churn-logistic-regression
