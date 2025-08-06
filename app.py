import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the trained model and scaler
model = joblib.load('model.pkl')  
scaler = joblib.load('scaler.pkl')

# Title
st.title("🏦 Loan Prediction App")

# Sidebar inputs
st.sidebar.header("📝 Applicant Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
loan_term = st.sidebar.selectbox("Loan Term (months)", [360, 180, 120, 60])
credit_history = st.sidebar.selectbox("Credit History", [1.0, 0.0])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Convert to DataFrame for prediction
input_data = pd.DataFrame({
    'Gender': [gender],
    'Married': [married],
    'Dependents': [dependents],
    'Education': [education],
    'Self_Employed': [self_employed],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_term],
    'Credit_History': [credit_history],
    'Property_Area': [property_area]
})

# Preprocessing function
def preprocess(df):
    df = df.copy()
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
    df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
    df['Property_Area'] = df['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
    df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)
    return df

input_processed = preprocess(input_data)

# Visualize user financial inputs
st.subheader("📊 Applicant Financial Summary")

# Create bar chart
fig, ax = plt.subplots()
bars = ax.bar(
    ['Applicant Income', 'Coapplicant Income', 'Loan Amount'],
    [applicant_income, coapplicant_income, loan_amount],
    color=['skyblue', 'orange', 'green']
)

# Add value labels
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f'{yval:.0f}', ha='center', va='bottom')

ax.set_ylabel("Amount")
ax.set_title("Income & Loan Overview")

st.pyplot(fig)

# Predict button
if st.button("🔍 Predict Loan Approval"):
    prediction = model.predict(input_processed)[0]
    if prediction == 'Y':
        st.success("✅ Loan will be Approved!")
    else:
        st.error("❌ Loan will be Rejected.")

