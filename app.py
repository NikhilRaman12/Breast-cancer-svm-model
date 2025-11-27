import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("svm_breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Breast Cancer Prediction (SVM Classifier)")
st.write("Upload a CSV file containing the patient feature values (without diagnosis).")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    # Read uploaded data
    input_data = pd.read_csv(uploaded_file)

    st.write("### Uploaded Data")
    st.dataframe(input_data)

    # Scale input
    data_scaled = scaler.transform(input_data)

    # Predict
    preds = model.predict(data_scaled)
    probs = model.predict_proba(data_scaled)[:, 1]

    # Prepare results
    results = pd.DataFrame({
        "Prediction (0=Benign,1=Malignant)": preds,
        "Cancer_Probability": probs
    })

    st.success("Prediction Completed!")
    st.write("### Results")
    st.dataframe(results)

    # Download results
    st.download_button(
        label="Download Predictions as CSV",
        data=results.to_csv(index=False),
        file_name="breast_cancer_svm_predictions.csv",
        mime="text/csv"
    )

st.markdown("---")
st.write("**Model:** SVM (RBF Kernel) | **Dataset:** Breast Cancer | **Developer:** Nikhil Raman – Data Scientist (AI/ML)")
