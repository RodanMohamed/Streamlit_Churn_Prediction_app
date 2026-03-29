import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model & scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'> Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.write("Predict whether a customer will churn based on their data.")

# Sidebar Inputs
st.sidebar.header(" Customer Information")

age = st.sidebar.slider("Age", 18, 80, 30)
tenure = st.sidebar.slider("Tenure (Months)", 0, 60, 12)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
gender = 0 if gender == "Male" else 1

# Arrange input
input_data = np.array([[age, tenure, gender]])

# Apply scaling only on Age & Tenure
input_data[:, [0,1]] = scaler.transform(input_data[:, [0,1]])

# Predict
if st.button(" Predict"):

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    st.subheader(" Prediction Result")

    if prediction == 1:
        st.error(" Customer is likely to CHURN")
    else:
        st.success(" Customer will stay")

    st.subheader(" Probability")

    st.write(f"Stay: {proba[0]:.2%}")
    st.write(f"Churn: {proba[1]:.2%}")

    # Progress bar
    st.progress(float(proba[1]))
