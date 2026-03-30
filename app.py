import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("fraud_model.pkl")

st.title("💳 Credit Card Fraud Detection")

st.write("Enter transaction details:")

# Input fields
amount = st.number_input("Transaction Amount", min_value=0.0)

features = []
for i in range(1, 29):
    val = st.number_input(f"V{i}", value=0.0)
    features.append(val)

if st.button("Predict"):
    input_data = np.array(features + [amount]).reshape(1, -1)
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")
