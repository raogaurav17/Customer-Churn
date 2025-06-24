import streamlit as st
import joblib
import numpy as np

st.title("Churn Prediction App")

st.divider()

st.write("This app predicts customer churn using a pre-trained model.")

st.divider()

age = st.number_input("Enter Age", min_value=18, max_value=100, value=30)

gender = st.selectbox("Enter the Gender", ["Male", "Female"])

tenure = st.number_input(
    "Enter Tenure (in months)", min_value=0, max_value=72, value=12
)

monthly_charges = st.number_input(
    "Enter Monthly Charges", min_value=30.0, max_value=200.0, value=50.0
)

st.divider()

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

predict_button = st.button("Predict", key="predict_button")

if predict_button:
    gender_selected = 1 if gender == "Female" else 0

    x = np.array([age, gender_selected, tenure, monthly_charges])

    x = scaler.transform([x])
    prediction = model.predict(x)

    predicted = "Churn" if prediction == 1 else "No Churn"

    st.write(f"The model predicts: **{predicted}**")

    proba = model.predict_proba(x)[0][1]  # Probability of churn
    st.write(f"Churn Probability: **{proba:.2%}**")


else:
    st.write("Click the button to make a prediction.")
