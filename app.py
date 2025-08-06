import streamlit as st
import pandas as pd
import numpy as np
import joblib  # or pickle

# Load the trained model
model = joblib.load('model.pkl')  
scaler = joblib.load('scaler.pkl')

# Title
st.title("Loan Prediction App")

# Sidebar inputs
st.sidebar.header("Applicant Information")

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

# Preprocessing should match the training preprocessing
# You must apply the same label encoding / one-hot encoding etc. here
# Example:
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

# Predict button
if st.button("Predict Loan Approval"):
    prediction = model.predict(input_processed)[0]
    if prediction == 'Y':
        st.success("✅ Loan will be Approved!")
    else:
        st.error("❌ Loan will be Rejected.")

# Load the historical dataset for visualization
@st.cache_data
def load_data():
    return pd.read_csv("loan_data.csv")  # Make sure this file exists

loan_df = load_data()

st.markdown("## 📊 Loan Data Insights")

# 1. Loan Status count
st.subheader("Loan Approval Distribution")
status_count = loan_df['Loan_Status'].value_counts()
st.bar_chart(status_count)

# 2. Loan Amount Distribution
st.subheader("Loan Amount Distribution")
st.hist_chart(loan_df['LoanAmount'].dropna())

# 3. Loan Status by Property Area
st.subheader("Loan Status by Property Area")
prop_status = loan_df.groupby(['Property_Area', 'Loan_Status']).size().unstack()
st.bar_chart(prop_status)

# 4. Loan Status by Dependents
st.subheader("Loan Status by Dependents")
dep_status = loan_df.groupby(['Dependents', 'Loan_Status']).size().unstack()
st.bar_chart(dep_status)
