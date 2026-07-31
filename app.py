import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set up the page configuration
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        return joblib.load('Random Forest Classifier.pkl')
    except FileNotFoundError:
        return None

model = load_model()

# Header Section
st.title("🫀 Heart Disease Risk Assessment Dashboard")
st.markdown("""
This application uses a Random Forest Classifier to predict the likelihood of heart disease based on clinical parameters. 
Please enter the patient's diagnostic metrics below.
""")
st.divider()

if model is None:
    st.error("Error: Model file 'rf_heart_disease_model.pkl' not found. Please ensure it is in the same directory as this script.")
else:
    # Create an attractive layout using columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Patient Demographics")
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        
        sex_display = st.selectbox("Sex", options=["Male", "Female"])
        sex = 1 if sex_display == "Male" else 0
        
        cp_display = st.selectbox("Chest Pain Type", options=[
            "Typical Angina (1)", 
            "Atypical Angina (2)", 
            "Non-anginal Pain (3)", 
            "Asymptomatic (4)"
        ])
        cp = int(cp_display[-2]) # Extracts the number from the string
        
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)

    with col2:
        st.subheader("🩸 Blood Work & ECG")
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        
        fbs_display = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=["No", "Yes"])
        fbs = 1 if fbs_display == "Yes" else 0
        
        restecg_display = st.selectbox("Resting ECG Results", options=[
            "Normal (0)", 
            "ST-T Wave Abnormality (1)", 
            "Left Ventricular Hypertrophy (2)"
        ])
        restecg = int(restecg_display[-2])
        
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)

    with col3:
        st.subheader("🏃 Stress Test Results")
        exang_display = st.selectbox("Exercise Induced Angina?", options=["No", "Yes"])
        exang = 1 if exang_display == "Yes" else 0
        
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        
        slope_display = st.selectbox("Slope of Peak Exercise ST Segment", options=[
            "Upsloping (1)", 
            "Flat (2)", 
            "Downsloping (3)"
        ])
        slope = int(slope_display[-2])
        
        ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 3, 0)
        
        thal_display = st.selectbox("Thalassemia (Thal)", options=[
            "Normal (3)", 
            "Fixed Defect (6)", 
            "Reversable Defect (7)"
        ])
        thal = int(thal_display[-2])

    st.divider()

    # Prediction Button
    if st.button("Run Diagnostic Analysis", type="primary", use_container_width=True):
        # Create a dataframe for the input to match training format
        input_data = pd.DataFrame([[
            age, sex, cp, trestbps, chol, fbs, restecg, 
            thalach, exang, oldpeak, slope, ca, thal
        ]], columns=[
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ])

        # Get prediction and probabilities
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        # Display results in a highly visible card
        st.subheader("🩺 Diagnostic Result")
        
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            if prediction == 1:
                st.error("⚠️ **High Risk of Heart Disease Detected**")
                st.markdown("The model indicates a significant likelihood of heart disease presence based on the provided clinical metrics.")
            else:
                st.success("✅ **Low Risk / No Heart Disease Detected**")
                st.markdown("The model indicates a low likelihood of heart disease presence.")
                
        with res_col2:
            st.metric(label="Model Confidence", value=f"{max(prediction_proba) * 100:.1f}%")