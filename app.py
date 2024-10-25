import streamlit as st
import pickle
import numpy as np

# Load the model and scaler
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Social Media Ads Engagement Classifier")

st.write("This app predicts if a user will engage with a social media ad based on their demographics.")

# Input fields for user data
age = st.slider("Age", 18, 65)
gender = st.selectbox("Gender", ["Male", "Female"])
estimated_salary = st.number_input("Estimated Annual Salary (in $)", 10000, 150000)

# Map gender to numerical values for model input
gender = 0 if gender == "Male" else 1

# Prepare input data for prediction
input_data = np.array([[age, gender, estimated_salary]])
input_data_scaled = scaler.transform(input_data)  # Apply scaling

# Make prediction
if st.button("Predict Engagement"):
    prediction = model.predict(input_data_scaled)
    result = "Engaged" if prediction == 1 else "Not Engaged"
    st.write(f"Prediction: The user will {result} with the ad.")
