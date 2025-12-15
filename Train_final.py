import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve,ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

#load dataset
df_raw = pd.read_csv(r"D:\\Employee Attrition\Employee-Attrition.csv")
df_raw=df_raw.drop(columns=['EmployeeCount', 'Over18', 'StandardHours'], axis=1)


# Create a copy for Training (We will encode this one)
df=df_raw.copy()
   
#Encode categorical variables
for col in df.select_dtypes(include=['object']).columns:
     le=LabelEncoder()
     df[col]=le.fit_transform(df[col])
     
#separating features and target 
  
X=df.drop('Attrition', axis=1)#features
y=df['Attrition']#target

#find top 10 features using RandomForest
#I am training a temporary model to find out which columns are actually important.
#from sklearn.ensemble import RandomForestClassifier
# rf_temp= RandomForestClassifier(n_estimators=100, random_state=42)
# rf_temp.fit(X, y)

# #creating a dataframe to see the importance of each feature
# feat_df=pd.DataFrame({
#      'Feature': X.columns, 
#      'Importance': rf_temp.feature_importances_   
# })

# #sorting them  to get important features
# best_features=feat_df.sort_values(by='Importance', ascending=False).head(10)['Feature'].tolist()
# print("selected features:", best_features)

best_features = [   
    'Age',
    'MonthlyIncome', 
    'OverTime', 
    'JobSatisfaction', 
    'YearsAtCompany', 
    'DistanceFromHome', 
    'TotalWorkingYears', 
    'YearsInCurrentRole',
    'JobInvolvement',   
    'MaritalStatus' ,
    'JobRole',          
    'StockOptionLevel'  
]

print("Using Custom Feature List:", best_features)

X_final = X[best_features]
#training
#now using only the best features to train the final model
X_train,X_test, y_train,y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

#scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train WITHOUT SMOTE (Imbalanced)
model_imbalanced = RandomForestClassifier(n_estimators=100, random_state=42)
model_imbalanced.fit(X_train_scaled, y_train)
predictions_imbalanced = model_imbalanced.predict(X_test_scaled)

ConfusionMatrixDisplay.from_predictions(y_test, predictions_imbalanced,display_labels=['Stay', 'Leave'], cmap='Blues')
print("\nConfusion Matrix without SMOTE:")
plt.title(" Imbalanced Model\n(Misses the Leavers)")
plt.show()

#handling class imbalance using SMOTE
#Because 'Yes' cases are very less compared to 'No'
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print("\n🏆 STARTING MODEL TRAINING...")

# List of models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

results = []
best_model_name = ""
best_recall = 0
final_model = None

for name, model in models.items():
    # Train
    model.fit(X_train_smote, y_train_smote)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate Scores
    rec = recall_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"   -> {name}: Recall={rec:.2f}, AUC={auc:.2f}")
    
    results.append({"Model": name, "Recall": rec, "Accuracy": acc, "AUC": auc})
    
    #check for best recall
    if rec > best_recall:
        best_recall = rec
        best_model_name = name
        final_model = model

results_df = pd.DataFrame(results).sort_values(by="Recall", ascending=False)

print("\n FINAL MODEL:")
print("="*60)
print(results_df)
print("="*60)
print(f" THE BEST MODEL IS: {best_model_name} (Recall: {best_recall:.2f})")

# Save the Comparison Table for Report
results_df.to_csv("Model_Comparison.csv", index=False)

#saving test results in csv form

print(f"Generating App Data using {best_model_name}...")

preds=final_model.predict(X_test_scaled)
probs=final_model.predict_proba(X_test_scaled)[:,1]

# 1. df_raw.loc[X_test.index] -> Gets the original rows (with text like 'Married')
# 2. [best_features] -> Keeps ONLY the 12 columns you want
results_table = df_raw.loc[X_test.index][best_features].copy()

results_table['ActualAttrition'] = y_test
results_table['PredictedAttrition'] = preds
results_table['risk_score(%)'] = (probs * 100).round(2)

#mapping back 'Yes'/'No' for better readability
results_table['ActualAttrition'] = results_table['ActualAttrition'].map({1: 'Yes', 0: 'No'})
results_table['PredictedAttrition'] = results_table['PredictedAttrition'].map({1: 'Yes', 0: 'No'})

#saving to csv
results_table.to_csv('test_results_selected_features.csv', index=False)
print("Created 'test_results.csv' with only 12 features!")

ConfusionMatrixDisplay.from_predictions(y_test, preds,display_labels=['Stay', 'Leave'], cmap='Blues')
print("\nConfusion Matrix with SMOTE:")
plt.title(f"Confusion Matrix ({best_model_name})")
plt.savefig("Confusion_Matrix.png", dpi=300)
plt.show()

#Final model after comparing both

acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds)
rec = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
roc_auc = roc_auc_score(y_test, probs)

print("\n" + "="*30)
print("\nFINAL MODEL EVALUATION REPORT:")
plt.title(f"FINAL MODEL EVALUATION REPORT({best_model_name})")
print("="*30)
print(f"1. Accuracy:   {acc:.2f}  (Overall Correctness)")
print(f"2. Precision:  {prec:.2f}  (Accuracy of 'Leaver' predictions)")
print(f"3. Recall:     {rec:.2f}  (Ability to catch all Leavers)")
print(f"4. F1-Score:   {f1:.2f}  (Balance of Precision & Recall)")
print(f"5. AUC-ROC:    {roc_auc:.2f} (Model's Distinguishing Power)")
print("="*30)

#plotting ROC curve
RocCurveDisplay.from_predictions(y_test, probs)
plt.title("ROC Curve")
plt.title(f"ROC Curve ({best_model_name})")
plt.savefig("ROC_Curve.png", dpi=300)
print("✅ ROC Graph Saved!")

#saving the files
#Saving model, scaler, and the list of features to use in the dashboard
joblib.dump(final_model,'attrition_model.pkl')
joblib.dump(scaler,'scaler.pkl')
joblib.dump(best_features,'features_list.pkl')

print(f"\n SUCCESS! Saved {best_model_name} as the final model.")
