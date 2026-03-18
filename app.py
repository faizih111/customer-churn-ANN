import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("model.h5")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Customer Churn App", layout="wide")

st.title("💼 Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn using your ANN model.")

# Sidebar inputs
st.sidebar.header("Input Customer Data")
credit_score = st.sidebar.slider("Credit Score", 300, 900, 600)
age = st.sidebar.slider("Age", 18, 100, 40)
tenure = st.sidebar.slider("Tenure (years)", 0, 10, 3)
balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 50000.0)
num_products = st.sidebar.slider("Number of Products", 1, 5, 1)
has_cr_card = st.sidebar.selectbox("Has Credit Card?", [0, 1])
is_active_member = st.sidebar.selectbox("Is Active Member?", [0, 1])
estimated_salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)
geography_germany = st.sidebar.selectbox("Geography Germany", [0, 1])
geography_spain = st.sidebar.selectbox("Geography Spain", [0, 1])
gender_male = st.sidebar.selectbox("Gender Male", [0, 1])

# Button
if st.button("Predict"):
    features = np.array(
        [
            [
                credit_score,
                age,
                tenure,
                balance,
                num_products,
                has_cr_card,
                is_active_member,
                estimated_salary,
                geography_germany,
                geography_spain,
                gender_male,
            ]
        ]
    )
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    st.subheader("Prediction Result:")
    if prediction[0][0] > 0.5:
        st.error("⚠️ Customer is likely to churn!")
    else:
        st.success("✅ Customer is likely to stay!")
