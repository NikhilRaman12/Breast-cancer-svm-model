import streamlit as st
import pandas as pd
import joblib
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(filename='prediction_log.txt', level=logging.INFO)

# Load model, scaler, and expected features
model = joblib.load("svm_breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_features = joblib.load("feature_names.pkl")

# App title
st.title("Breast Cancer Prediction (SVM Classifier)")
st.write("Upload a CSV file containing the patient feature values (without diagnosis).")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    input_data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data")
    st.dataframe(input_data)

    # Validate columns
    missing = [col for col in expected_features if col not in input_data.columns]
    extra = [col for col in input_data.columns if col not in expected_features]

    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()
    elif extra:
        st.warning(f"Extra columns detected: {extra}. They will be ignored.")
        input_data = input_data[expected_features]
    else:
        input_data = input_data[expected_features]

    # Scale and predict
    try:
        data_scaled = scaler.transform(input_data)
        preds = model.predict(data_scaled)
        probs = model.predict_proba(data_scaled)[:, 1]

        results = pd.DataFrame({
            "Prediction (0=Benign,1=Malignant)": preds,
            "Cancer_Probability": probs
        })

        # Log predictions
        logging.info(f"Prediction run at {datetime.now()}")
        logging.info(f"Input shape: {input_data.shape}")
        logging.info(f"Predictions: {preds.tolist()}")
        logging.info(f"Probabilities: {probs.tolist()}")

        # Show results
        st.success("Prediction Completed!")
        st.write("### Results")
        st.dataframe(results)

        # Confidence threshold interpretation
        high_risk = results[results["Cancer_Probability"] > 0.85]
        st.write(f"High-risk cases detected: {len(high_risk)} (Probability > 0.85)")

        # Visualization
        fig, ax = plt.subplots()
        ax.hist(probs, bins=10, color='skyblue')
        ax.set_title("Cancer Probability Distribution")
        ax.set_xlabel("Probability")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        # Download button
        st.download_button(
            label="Download Predictions as CSV",
            data=results.to_csv(index=False),
            file_name="breast_cancer_svm_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Footer with version info
version = "v1.0.0"
last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

st.markdown("---")
st.write(f"Model: SVM (RBF Kernel) | Version: {version} | Last Updated: {last_updated}")
st.write("Dataset: Breast Cancer | Developer: Nikhil Raman – Data Scientist (AI/ML)")