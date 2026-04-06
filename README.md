# 🏢 Employee Attrition Analysis & Prediction System

---

## 📌 Project Introduction

Employee turnover (attrition) is a major challenge for HR departments, leading to increased hiring costs and loss of institutional knowledge. This project aims to solve this problem by analyzing historical employee data to identify the root causes of attrition and building Machine Learning models to:

1. **Predict which employees are at "High Risk" of leaving** (Classification)
2. **Predict how long an employee has gone without a promotion** — a key driver of attrition (Regression)

The system is deployed as an interactive **Streamlit Dashboard** that allows HR managers to visualize workforce insights and run real-time predictions for individual employees.

---

## 🛠️ Technologies & Tools Used

| Category | Technology / Library | Usage |
|---|---|---|
| Language | Python 3.9+ | Core programming language |
| Data Manipulation | Pandas, NumPy | Data cleaning, transformation, and array operations |
| Visualization | Matplotlib, Seaborn | Exploratory Data Analysis (EDA) and plotting (Heatmaps, Boxplots) |
| Machine Learning | Scikit-Learn | Model training, scaling, and evaluation metrics |
| Imbalance Handling | Imbalanced-learn (SMOTE) | Generating synthetic data to fix class imbalance |
| Web Framework | Streamlit | Building the interactive web dashboard |
| Model Saving | Joblib | Saving/Loading the trained model (.pkl files) |

---

## ⚙️ Technical Architecture & Workflow

### Task 1: Attrition Prediction (Classification)

#### 1. Data Cleaning & Preprocessing
- **Feature Removal:** Dropped columns with zero variance (`EmployeeCount`, `Over18`, `StandardHours`) as they provide no predictive value.
- **Outlier Treatment:** Applied IQR (Interquartile Range) Capping to handle extreme values in columns like `MonthlyIncome` to prevent model skewing.
- **Encoding:** Converted categorical variables (e.g., `BusinessTravel`, `Department`) into numeric formats using Label Encoding.
- **Scaling:** Applied `StandardScaler` to normalize continuous features (`Age`, `Income`, `YearsAtCompany`) so that features with larger ranges don't dominate the model.

#### 2. Feature Selection (Hybrid Approach)
Instead of relying solely on the machine or using all 30+ columns, we used a **Hybrid Selection Strategy** combining statistical analysis with human domain expertise:

- **Machine Preference (Statistical):** A Random Forest Classifier was used to mathematically identify the strongest predictors, such as `MonthlyIncome`, `OverTime`, and `Age`.
- **Human Understanding (Domain Expertise):** Behavioral features like `JobSatisfaction`, `JobInvolvement`, and `DistanceFromHome` were manually prioritized. While a machine might overlook these in favor of raw numbers, human intuition suggests these are critical psychological drivers for an employee deciding to quit.

**Final 12 Selected Features:** `Age`, `MonthlyIncome`, `OverTime`, `JobSatisfaction`, `YearsAtCompany`, `DistanceFromHome`, `TotalWorkingYears`, `YearsInCurrentRole`, `JobInvolvement`, `MaritalStatus`, `JobRole`, `StockOptionLevel`

#### 3. Handling Class Imbalance
The original dataset was highly imbalanced (~84% Stay vs. ~16% Leave). **SMOTE** (Synthetic Minority Over-sampling Technique) was applied on the training data to generate synthetic examples of "Leavers," ensuring the model doesn't bias towards the majority class.

#### 4. Model Training & Comparison
Five different algorithms were trained to find the best performer:
- Logistic Regression (Baseline)
- Decision Tree
- Random Forest (Ensemble)
- Gradient Boosting (Ensemble)
- Support Vector Machine (SVM)

#### 5. Evaluation Metric
The model was optimized for **Recall**.

> **Why Recall?** In employee attrition, a False Negative (predicting an employee will stay, but they leave) is the most expensive error. We want to catch as many potential leavers as possible.

---

### Task 3: Predicting Employee Promotion Likelihood (Regression)

#### Problem Statement
> *"Can we predict how many years it has been since an employee's last promotion — and use that to identify who is overdue for one?"*

- **Target Variable:** `YearsSinceLastPromotion` (continuous, range 0–10 years)
- **Problem Type:** Supervised Regression
- **Business Value:** Employees who are high performers but have gone many years without a promotion are a hidden attrition risk — the model flags these individuals for HR action.

#### 1. Data Preparation
- **Outlier Removal:** Removed 93 employees with `YearsSinceLastPromotion > 10` (extreme edge cases that skewed training). Dataset reduced from 1,470 → 1,377 rows.
- **Encoding & Scaling:** Same pipeline as Task 1 — Label Encoding for categoricals, `StandardScaler` for all features.
- **Train/Test Split:** 80% training / 20% test (`random_state=42`).

#### 2. Feature Selection
Top 11 features were selected using Random Forest feature importance scores:

| Rank | Feature | Importance Score |
|---|---|---|
| 1 | YearsInCurrentRole | **0.690** — strongest signal |
| 2 | YearsWithCurrManager | 0.502 |
| 3 | JobLevel | 0.085 |
| 4 | Education | 0.051 |
| 5 | TotalWorkingYears | 0.044 |
| 6 | YearsAtCompany | 0.043 |
| 7 | PerformanceRating | 0.039 |
| 8 | JobInvolvement | 0.027 |
| 9 | WorkLifeBalance | 0.019 |
| 10 | NumCompaniesWorked | 0.017 |
| 11 | OverTime | 0.012 |

> **Key Finding:** How long an employee has been in their current role is by far the strongest predictor of promotion timing.

#### 3. Model Training & Comparison
Four regression algorithms were trained and evaluated:

| Model | MAE | RMSE | R² | Verdict |
|---|---|---|---|---|
| Linear Regression | 1.371 yrs | 1.973 yrs | 0.258 | Baseline |
| Decision Tree | 1.555 yrs | 2.645 yrs | −0.334 | Overfit — rejected |
| Random Forest | 1.287 yrs | 1.967 yrs | 0.263 | Good |
| **Gradient Boosting** | **1.307 yrs** | **1.947 yrs** | **0.277** | **Best — selected ✅** |

**Metrics Explained:**
- **MAE** — average error in years (lower = better)
- **RMSE** — penalises large errors more (lower = better)
- **R²** — proportion of variance the model explains (higher = better)

> **Note:** Decision Tree had a negative R² (−0.334), meaning it performed worse than simply predicting the mean — indicating severe overfitting.

#### 4. Final Model — Gradient Boosting Regressor

**Configuration:** `n_estimators=500`, `max_depth=3`, `learning_rate=0.05`

| Metric | Score | Interpretation |
|---|---|---|
| MAE | **1.307 years** | Predictions off by ~1.3 years on average |
| RMSE | **1.947 years** | Within ~2 years on a 0–10 scale |
| R² | **0.277** | Captures 27.7% of variance in promotion timing |

**Why Gradient Boosting?**
- Builds trees sequentially — each tree corrects the previous one's errors
- Shallow trees (`max_depth=3`) prevent overfitting
- Outperformed all other models on RMSE and R²

#### 5. Business Interpretation

| Predicted Years Since Promotion | Interpretation | HR Action |
|---|---|---|
| 0–2 years | Recently promoted or due soon | No immediate action |
| 3–5 years | Approaching overdue | Schedule career conversation |
| 6–10 years | **Seriously overdue** | **High attrition risk — prioritise for promotion review** |

> **Cross-link with Task 1:** Employees with a high predicted `YearsSinceLastPromotion` AND a high `PerformanceRating` appear in both models as attrition risks — giving HR a complete, dual-signal risk picture.

#### 6. Limitations
- **R² = 0.277 is intentionally moderate** — promotion decisions involve management discretion, budgets, and organizational politics that are not captured in any dataset.
- The model captures structural, measurable patterns (role tenure, manager tenure); it cannot predict subjective human decisions.

---

## 📂 Project Structure

```
Employee Attrition/
├── app.py                              # Streamlit Dashboard (4 tabs)
├── Train_final.py                      # Task 1: Cleaning → SMOTE → Classification → Saving
├── Train_regression.py                 # Task 3: Cleaning → Regression → Saving
├── README.md                           # Project documentation
│
├── data/
│   ├── raw/
│   │   └── Employee-Attrition.csv      # The raw HR dataset (1,470 employees)
│   └── processed/
│       ├── Employee_Attrition_Scaled.csv       # Scaled dataset
│       ├── Model_Comparison.csv                # Performance report — 5 classification models
│       └── test_results_selected_features.csv  # Test set predictions with risk scores
│
├── models/
│   ├── attrition_model.pkl             # Best trained classification model (Task 1)
│   ├── scaler.pkl                      # Saved Scaler for attrition model
│   ├── features_list.pkl               # 12 selected features for attrition model
│   ├── promotion_model.pkl             # Best trained regression model (Task 3)
│   ├── promotion_scaler.pkl            # Saved Scaler for promotion model
│   └── promotion_features.pkl          # 11 selected features for promotion model
│
├── outputs/
│   ├── Confusion_Matrix.png            # Confusion matrix plot
│   └── ROC_Curve.png                   # ROC curve plot
│
└── notebooks/
    ├── attrition.ipynb                 # EDA & preprocessing notebook
    └── emp_attrition.ipynb             # Feature engineering notebook
```

---

## 🚀 How to Run the Project

### Step 1: Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn streamlit joblib
```

### Step 2: Train the Attrition Model (Task 1)
```bash
python Train_final.py
```
Evaluation scores for all 5 classification models will be displayed, and the best model will be saved to `models/`.

### Step 3: Train the Promotion Model (Task 3)
```bash
python Train_regression.py
```
Evaluation scores for all 4 regression models will be displayed, and the best model will be saved to `models/`.

### Step 4: Run the Dashboard
```bash
streamlit run app.py
```

The dashboard has 4 tabs:
| Tab | Description |
|---|---|
| 📉 Attrition Report | Company-wide attrition analytics and charts |
| 🤝 Diversity Report | Workforce diversity breakdown |
| 🔮 Prediction Tool | Real-time attrition risk prediction for an employee |
| 📈 Promotion Predictor | Real-time promotion timeline prediction (Task 3) |

---

## 📢 Actionable Insights for HR (Strategic Recommendations)

Based on the data analysis and model predictions, the following strategies are recommended to improve employee retention:

### 1. Tackle the "OverTime" Issue
- **Insight:** Employees working frequent overtime are ~3× more likely to leave (~30.5% attrition vs. 10.4% for non-overtime employees).
- **Action:** Conduct a workload audit. If a team is constantly on overtime, consider hiring additional support or redistributing tasks. Implement "No-Meeting Fridays" to reduce burnout.

### 2. Review Compensation Structures
- **Insight:** Lower `MonthlyIncome` was a top predictor of attrition, especially in early-career roles.
- **Action:** Benchmark salaries against industry standards. Consider introducing performance-based bonuses or stock options (`StockOptionLevel` was also a key feature) for high performers who are at risk.

### 3. Focus on Career Growth for Young Employees
- **Insight:** Younger employees (Age < 30) and those with fewer `YearsAtCompany` have higher turnover rates. Most attrition happens in the first 0–3 years.
- **Action:** Implement a structured mentorship program. Create clear "Career Pathing" maps so junior employees can visualize their future growth within the company rather than looking elsewhere.

### 4. Improve Job Satisfaction
- **Insight:** `JobSatisfaction` and `EnvironmentSatisfaction` scores directly correlate with retention.
- **Action:** Run anonymous quarterly pulse surveys to understand why satisfaction is low. Address "quick wins" like office environment improvements or flexible working hours (`WorkLifeBalance`).

### 5. Proactively Manage Promotions (Task 3 Finding)
- **Insight:** `YearsInCurrentRole` is the strongest predictor of promotion timing (importance: 0.690). Employees stuck in the same role for 6+ years are a significant attrition risk — especially high performers.
- **Action:** Use the **Promotion Predictor tab** in the dashboard to identify overdue employees. Cross-reference with attrition risk scores from the Prediction Tool to prioritise the highest-risk individuals for promotion review cycles.
