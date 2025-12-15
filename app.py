import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="HR Analytics", layout="wide")

st.title("🏢 HR Analytics & Prediction System")
#load the saved files
try:
     model = joblib.load('attrition_model.pkl')
     scaler = joblib.load('scaler.pkl')
     features = joblib.load('features_list.pkl')
     df_raw = pd.read_csv(r"D:\Employee Attrition\Employee-Attrition.csv")
     
     # Load the Test Results (The Report Card)
     test_data = pd.read_csv(r"D:\Employee Attrition\test_results_selected_features.csv")
     leaderboard = pd.read_csv(r"D:\Employee Attrition\Model_Comparison.csv")
except:
    st.error("Files not found! Please run 'train_final.py' first.")
    st.stop()

#create tabs

tab1, tab2, tab3 = st.tabs(["📉 Attrition Report", "🤝 Diversity Report", "🔮 Prediction Tool"])

with tab1:
    st.header("Company Overview")
    
    # --- STEP 1: DO THE MATH FIRST ---
    # 1. Count total people
    total_employees = len(df_raw)
    
    # 2. Count people who said 'Yes' to Attrition
    leavers_data =df_raw[df_raw['Attrition'] == 'Yes']
    total_leavers = len(leavers_data)
    
    # 3. Calculate percentage
    # (Leavers divided by Total) multiplied by 100
    attrition_rate = (total_leavers / total_employees) * 100
    
    # 4. Calculate Average Salary
    avg_salary = df_raw['MonthlyIncome'].mean()
    
    
    
    # --- STEP 2: SHOW THE NUMBERS (METRICS) ---
    # Create 3 columns side-by-side
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Total Employees", total_employees)
    # round(number, 1) means keep 1 decimal place (e.g., 16.1)
    col2.metric("Attrition Rate", f"{round(attrition_rate, 1)}%") 
    col3.metric("Avg Income", f"${round(avg_salary)}")
    
    st.markdown("---") # Draws a line
    
    # --- STEP 3: DRAW THE CHART ---
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Attrition by Department")
        # Creating a simple figure
        fig1 = plt.figure(figsize=(6, 4))
        sns.countplot(data=df_raw, x='Department', hue='Attrition', palette='coolwarm')
        st.pyplot(fig1)
        
    with c2:
        st.subheader("Attrition by Job Role")
        # Only showing roles where people left
        fig2 = plt.figure(figsize=(6, 4))
        sns.countplot(data=leavers_data, y='JobRole', palette='Reds')
        st.pyplot(fig2)
        
        
with tab2:
    st.header("Workforce Diversity")
    
    # --- Step 1: Calculate Numbers ---
    # Count Men and Women
    gender_counts = df_raw['Gender'].value_counts()
    avg_age = df_raw['Age'].mean()
    
    # --- Step 2: Show Metrics ---
    d1, d2, d3 = st.columns(3)
    d1.metric("Average Age", f"{avg_age:.0f} Years")
    d1.caption("The average age of our workforce")
    
    d2.metric("Male Employees", gender_counts['Male'])
    d3.metric("Female Employees", gender_counts['Female'])
    
    st.markdown("---")
    
    # --- Step 3: Draw Charts ---
    g1, g2 = st.columns(2)
    
    with g1:
        st.subheader("Gender Distribution")
        fig3 = plt.figure(figsize=(6, 4))
        # Simple Pie Chart
        plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['lightblue', 'pink'])
        st.pyplot(fig3)
        
    with g2:
        st.subheader("Age Group Distribution")
        fig4 = plt.figure(figsize=(6, 4))
        sns.histplot(df_raw['Age'], kde=True, color='purple')
        st.pyplot(fig4)

with tab3:
    
    
    #model table
    
    st.header("1. Model Selection")
    st.write("compared 5 models to find the best one.")
    
    st.dataframe(leaderboard)
    
    winner_name = leaderboard.iloc[0]['Model']
    winner_score = leaderboard.iloc[0]['Recall']
    
    st.success(f"🏆 The final model is: **{winner_name}** (Recall: {winner_score:.2f})")
    st.caption("This model was automatically selected for the prediction below.")
    
    st.markdown("---")
    
    # --- PART A: MODEL VALIDATION (15 RECORDS) ---
    st.header("1. Test Data Verification")
    st.write("Here are 15 real examples to prove the model accuracy.")
    
    sample=test_data.head(15).copy()
    st.dataframe(sample)
    
    st.markdown("---")
        # --- PART B: USER INPUT FOR PREDICTION ---
    st.header("Predict Employee Risk")
    
    # Creating a container box
    with st.container():
        
        user_inputs = {}
        
        # Split screen into Left and Right
        col_left, col_right = st.columns(2)
        
        # --- LEFT SIDE: Personal Details ---
        with col_left:
            st.subheader("👤 Personal Details")
            
            # Age
            user_inputs['Age'] = st.number_input("Age", min_value=18, max_value=60, value=30)
            
            # Marital Status
            status = st.selectbox("Marital Status", ["Divorced", "Married", "Single"])
            # Simple If-Else conversion
            if status == "Divorced":
                user_inputs['MaritalStatus'] = 0
            elif status == "Married":
                user_inputs['MaritalStatus'] = 1
            else: # Single
                user_inputs['MaritalStatus'] = 2
            
            # Distance
            user_inputs['DistanceFromHome'] = st.slider("Distance From Home (km)", 1, 30, 5)
            
            # Satisfaction Sliders
            user_inputs['JobInvolvement'] = st.slider("Job Involvement (1-4)", 1, 4, 3)
            user_inputs['JobSatisfaction'] = st.slider("Job Satisfaction (1-4)", 1, 4, 2)

        # --- RIGHT SIDE: Work Details ---
        with col_right:
            st.subheader("💼 Job Details")
            
            # Income
            user_inputs['MonthlyIncome'] = st.number_input("Monthly Income", value=5000)
            
            # OverTime
            ot_choice = st.selectbox("OverTime?", ["No", "Yes"])
            if ot_choice == "Yes":
                user_inputs['OverTime'] = 1
            else:
                user_inputs['OverTime'] = 0
            
            # Job Role
            # Get list of roles from data
            roles_list = sorted(df_raw['JobRole'].unique())
            selected_role = st.selectbox("Job Role", roles_list)
            # Find the index (0, 1, 2...)
            user_inputs['JobRole'] = roles_list.index(selected_role)
            
            # Other Numbers
            user_inputs['YearsAtCompany'] = st.number_input("Years at Company", value=5)
            user_inputs['YearsInCurrentRole'] = st.number_input("Years in Current Role", value=2)
            user_inputs['TotalWorkingYears'] = st.number_input("Total Working Years", value=8)
            user_inputs['StockOptionLevel'] = st.slider("Stock Option Level (0-3)", 0, 3, 0)

        # --- PREDICT BUTTON ---
        st.markdown("---")
        if st.button("🚀 Analyze Risk", use_container_width=True):
            
            # 1. Convert dictionary to Table
            input_df = pd.DataFrame([user_inputs])
            
            # 2. Select columns in correct order (Important!)
            final_input = input_df[features]
            
            # 3. Scale Data
            scaled_input = scaler.transform(final_input)
            
            # 4. Predict
            prediction = model.predict(scaled_input)
            prob = model.predict_proba(scaled_input)
            
            # Calculate Confidence Score (0 to 100)
            risk_score = prob[0][1] * 100
            stay_score = prob[0][0] * 100
            
            
            
            # 5. Show Result
            st.subheader("Analysis Result:")
            
            if prediction[0] == 1:
                st.error(f"⚠️ HIGH RISK DETECTED (Confidence: {risk_score:.0f}%)")
                st.write(f"👉 **Risk Probability:** {risk_score:.0f}% (Very High)")
                st.write("**Reason:** Factors like **OverTime** or **Low Income** suggest this employee might leave.")
            else:
                st.success(f"✅ LOW RISK / SAFE (Confidence: {risk_score:.0f}%)")
                st.write(f"👉 **Chance of Staying:** {stay_score:.0f}% (High Stability)")
                st.write("**Reason:** Key factors suggest this employee is **stable and likely to stay**.")