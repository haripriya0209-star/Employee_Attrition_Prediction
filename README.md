# 🏢 Employee Attrition Analysis & Prediction System

## 📌 Project Introduction
Employee turnover (attrition) is a major challenge for HR departments, leading to increased hiring costs and loss of institutional knowledge. This project aims to solve this problem by analyzing historical employee data to identify the root causes of attrition and building a Machine Learning model to predict which employees are at "High Risk" of leaving.

The system is deployed as an interactive **Streamlit Dashboard** that allows HR managers to visualize workforce insights and run real-time predictions for individual employees.

---

## 🛠️ Technologies & Tools Used

| Category | Technology / Library | Usage |
| :--- | :--- | :--- |
| **Language** | Python 3.9+ | Core programming language |
| **Data Manipulation** | Pandas, NumPy | Data cleaning, transformation, and array operations |
| **Visualization** | Matplotlib, Seaborn | Exploratory Data Analysis (EDA) and plotting (Heatmaps, Boxplots) |
| **Machine Learning** | Scikit-Learn | Model training, scaling, and evaluation metrics |
| **Imbalance Handling** | Imbalanced-learn (SMOTE) | Generating synthetic data to fix class imbalance |
| **Web Framework** | Streamlit | Building the interactive web dashboard |
| **Model Saving** | Joblib | Saving/Loading the trained model (`.pkl` files) |

---

## ⚙️ Technical Architecture & Workflow

### 1. Data Cleaning & Preprocessing
- **Feature Removal:** Dropped columns with zero variance (`EmployeeCount`, `Over18`, `StandardHours`) as they provide no predictive value.
- **Outlier Treatment:** Applied **IQR (Interquartile Range) Capping** to handle extreme values in columns like `MonthlyIncome` to prevent model skewing.
- **Encoding:** Converted categorical variables (e.g., `BusinessTravel`, `Department`) into numeric formats using **Label Encoding**.
- **Scaling:** Applied **StandardScaler** to normalize continuous features (Age, Income, YearsAtCompany) so that features with larger ranges don't dominate the model.

### 2. Feature Selection (Hybrid Approach)
Instead of relying solely on the machine or using all 30+ columns, we used a **Hybrid Selection Strategy** combining statistical analysis with human domain expertise:

- **Machine Preference (Statistical):** A Random Forest Classifier was used to mathematically identify the strongest predictors, such as `MonthlyIncome`, `OverTime`, and `Age`.
- **Human Understanding (Domain Expertise):** Behavioral features like `JobSatisfaction`, `JobInvolvement`, and `DistanceFromHome` were manually prioritized. While a machine might overlook these in favor of raw numbers, human intuition suggests these are critical psychological drivers for an employee deciding to quit.

**Final 12 Selected Features:**
`Age`, `MonthlyIncome`, `OverTime`, `JobSatisfaction`, `YearsAtCompany`, `DistanceFromHome`, `TotalWorkingYears`, `YearsInCurrentRole`, `JobInvolvement`, `MaritalStatus`, `JobRole`, `StockOptionLevel`

### 3. Handling Class Imbalance
The original dataset was highly imbalanced (~84% Stay vs. ~16% Leave). **SMOTE (Synthetic Minority Over-sampling Technique)** was applied on the training data to generate synthetic examples of "Leavers," ensuring the model doesn't bias towards the majority class.

### 4. Model Training & Comparison
Five different algorithms were trained to find the best performer:

1. **Logistic Regression** (Baseline)
2. **Decision Tree**
3. **Random Forest** (Ensemble)
4. **Gradient Boosting** (Ensemble)
5. **Support Vector Machine (SVM)**

### 5. Evaluation Metric
The model was optimized for **Recall**.

> **Why Recall?** In employee attrition, a **False Negative** (predicting an employee will stay, but they leave) is the most expensive error. We want to catch as many potential leavers as possible.

---

## 📂 Project Structure

```text
Employee Attrition/
├── app.py                              # Streamlit Dashboard script
├── Train_final.py                      # Main script: Cleaning → SMOTE → Training → Saving
├── README.md                           # Project documentation
│
├── data/
│   ├── raw/
│   │   └── Employee-Attrition.csv      # The raw HR dataset
│   └── processed/
│       ├── Employee_Attrition_Scaled.csv   # Scaled dataset
│       ├── Model_Comparison.csv            # Performance report of all 5 models
│       └── test_results_selected_features.csv  # Test set predictions with risk scores
│
├── models/
│   ├── attrition_model.pkl             # Best trained model
│   ├── scaler.pkl                      # Saved Scaler for new data
│   └── features_list.pkl              # List of the 12 selected features
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

### Step 2: Train the Model
Run the training script to process the data and generate the model files:
```bash
python Train_final.py
```
Output: Evaluation scores for all 5 models will be displayed, and the best model will be automatically saved to the `models/` folder.

### Step 3: Run the Dashboard
Launch the web application:
```bash
streamlit run app.py
```

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
