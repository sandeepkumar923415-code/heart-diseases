import streamlit as st
import pandas as pd
import joblib 

# Load saved files
model = joblib.load("knn_heart.pkl")
scalar = joblib.load("scalar.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("Heart Disease Prediction by Sandip ❤️")
st.markdown("Provide the following details")

# Collect user input
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# When Predict is clicked
if st.button("Predict"):

    # Create dictionary with ALL expected columns set to 0
    input_dict = {col: 0 for col in expected_columns}

    # Fill numeric values
    input_dict['Age'] = age
    input_dict['RestingBP'] = resting_bp
    input_dict['Cholesterol'] = cholesterol
    input_dict['FastingBS'] = fasting_bs
    input_dict['MaxHR'] = max_hr
    input_dict['Oldpeak'] = oldpeak

    # Fill categorical encoded columns SAFELY
    if f'Sex_{sex}' in input_dict:
        input_dict[f'Sex_{sex}'] = 1

    if f'ChestPainType_{chest_pain}' in input_dict:
        input_dict[f'ChestPainType_{chest_pain}'] = 1

    if f'RestingECG_{resting_ecg}' in input_dict:
        input_dict[f'RestingECG_{resting_ecg}'] = 1

    if f'ExerciseAngina_{exercise_angina}' in input_dict:
        input_dict[f'ExerciseAngina_{exercise_angina}'] = 1

    if f'ST_Slope_{st_slope}' in input_dict:
        input_dict[f'ST_Slope_{st_slope}'] = 1

    # Convert to dataframe
    input_df = pd.DataFrame([input_dict])

    # Ensure correct column order
    input_df = input_df[expected_columns]

    # Scale input
    scaled_input = scalar.transform(input_df)

    # Prediction
    prediction = model.predict(scaled_input)[0]

    # Show result
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
