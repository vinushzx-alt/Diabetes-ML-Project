import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🩺 Diabetes Prediction Assistant")
st.write("Enter the patient's medical details below to predict the likelihood of diabetes.")

# Create input columns for a cleaner UI
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=79)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=32.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.000, max_value=3.000, value=0.500, format="%.3f")
    age = st.number_input("Age", min_value=1, max_value=120, value=33)

# Prediction Button
if st.button("Predict"):
    # Organize inputs into a numpy array
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Scale the input data using our saved scaler
    scaled_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1]

    st.markdown("---")
    if prediction[0] == 1:
        st.error(f"**High Risk of Diabetes.** (Probability: {probability * 100:.2f}%)")
        st.write("Please consult with a healthcare professional.")
    else:
        st.success(f"**Low Risk of Diabetes.** (Probability: {probability * 100:.2f}%)")
        st.write("Maintain a healthy lifestyle!")