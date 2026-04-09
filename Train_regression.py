import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# ─── Load & Clean Data ────────────────────────────────────────────────────────
df = pd.read_csv(r"data\raw\Employee-Attrition.csv")
df = df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours'])

# Encode categorical columns (same as notebook)
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Remove extreme outliers — employees with 11–15 years since last promotion
before = len(df)
df = df[df['YearsSinceLastPromotion'] <= 10]
print(f"Removed {before - len(df)} outlier rows (>10 years since promotion)")
print(f"Dataset size: {before} → {len(df)}")

# ─── Features (top 11 — selected from notebook experiments) ──────────────────
promotion_features = [
    'YearsInCurrentRole',    # impact: 0.690 — strongest signal
    'YearsWithCurrManager',  # impact: 0.502
    'JobLevel',              # impact: 0.085
    'Education',             # impact: 0.051
    'TotalWorkingYears',     # impact: 0.044
    'YearsAtCompany',        # impact: 0.043
    'PerformanceRating',     # impact: 0.039
    'JobInvolvement',        # impact: 0.027
    'WorkLifeBalance',       # impact: 0.019
    'NumCompaniesWorked',    # impact: 0.017
    'OverTime',              # impact: 0.012
]

X = df[promotion_features]
y = df['YearsSinceLastPromotion']

# ─── Split ────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─── Scale ────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─── Train ────────────────────────────────────────────────────────────────────
# Gradient Boosting was the best model in notebook experiments (RMSE=1.947, R²=0.277)
model = GradientBoostingRegressor(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)

print("\nTraining Gradient Boosting Regressor...")
model.fit(X_train_scaled, y_train)

# ─── Evaluate ─────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("\n" + "=" * 50)
print("REGRESSION MODEL EVALUATION:")
print("=" * 50)
print(f"  MAE  (Avg error in years):  {mae:.3f}")
print(f"  RMSE (Penalised error):     {rmse:.3f}")
print(f"  R²   (Pattern explained):   {r2:.3f}")
print("=" * 50)
print(f"  Target range: 0–10 years")
print(f"  RMSE < 2 means predictions are within ~2 years on average")

# ─── Save ─────────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

joblib.dump(model,              'models/promotion_model.pkl')
joblib.dump(scaler,             'models/promotion_scaler.pkl')
joblib.dump(promotion_features, 'models/promotion_features.pkl')

print("\nSaved:")
print("  models/promotion_model.pkl")
print("  models/promotion_scaler.pkl")
print("  models/promotion_features.pkl")
print("\nRun app.py to use the Promotion Prediction tab!")
