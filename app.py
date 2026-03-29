import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Churn Prediction", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
        border: none;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_models()
except:
    st.error("Model files not found. Please ensure 'churn_model.pkl' and 'scaler.pkl' are in the directory.")


col_t1, col_t2 = st.columns([1, 4])
with col_t1:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100) # Generic AI Icon
with col_t2:
    st.title("ChurnGuard AI")
    st.caption("Advanced Predictive Analytics for Customer Retention")

st.divider()


st.subheader("👤 Customer Profile")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 100, 30)
with col2:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
with col3:
    gender_raw = st.selectbox("Gender", ["Male", "Female"])
    gender = 0 if gender_raw == "Male" else 1


st.markdown("---")
if st.button(" Run Analysis"):
    # Prepare data
    input_data = np.array([[age, tenure, gender]])
    input_data[:, [0,1]] = scaler.transform(input_data[:, [0,1]])
    
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    
    # --- RESULTS DISPLAY ---
    st.subheader(" Analysis Results")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if prediction == 1:
            st.error("###  High Risk of Churn")
            st.write("This customer shows behavior patterns consistent with past churned users.")
        else:
            st.success("###  Low Risk / Loyal")
            st.write("This customer is likely to stay with the current service plan.")

    with res_col2:
       
        churn_risk = proba[1]
        st.metric(label="Churn Probability", value=f"{churn_risk:.1%}")
        st.progress(churn_risk)

    # --- INSIGHTS ---
    with st.expander("See Detailed Breakdown"):
        st.info(f"The model is {(max(proba)*100):.2f}% confident in this prediction.")
        
        chart_data = pd.DataFrame({"Feature": ["Age", "Tenure"], "Value": [age, tenure]})
        st.bar_chart(chart_data.set_index("Feature"))

else:
    st.info("Adjust the customer parameters above and click 'Run Analysis' to see results.")
