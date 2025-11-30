#  Breast Cancer Prediction using SVM (Streamlit Deployment)

An end-to-end Machine Learning project that builds, evaluates, and deploys an SVM-based breast cancer tumor classifier using the Breast Cancer Wisconsin dataset.  
This project demonstrates a complete ML workflow — from data preprocessing to real-time prediction through a Streamlit web application.

---

##  Project Highlights

- **Support Vector Machine (RBF Kernel)** for high-accuracy classification  
- **Complete preprocessing pipeline**: cleaning, encoding, scaling  
- **Hyperparameter tuning** for better model performance  
- **Interactive Streamlit web app** for real-time predictions  
- Upload patient feature CSV → get instant predictions  
- Model saved using **Joblib**, integrated into the app  
- Fully deployed and publicly accessible  

---

##  Live Demo

**Streamlit App:**  
https://breast-cancer-svm-model-6an2ymwthnprsvwjgckdcw.streamlit.app/

---

##  Repository Structure

```
|-- app.py                     # Streamlit Web App
|-- model.py                   # Model training script
|-- breast-cancer.csv          # Dataset
|-- patient_features.csv       # Sample input format for app users
|-- svm_breast_cancer_model.pkl
|-- scaler.pkl
|-- requirements.txt
|-- README.md
```

---

## Machine Learning Workflow

1. **Load Dataset**  
2. **Preprocessing**  
   - Handle missing values  
   - Remove non-feature columns  
   - Apply StandardScaler  
3. **Model Training**  
   - SVM (RBF Kernel)  
   - Hyperparameter tuning  
4. **Model Evaluation**  
   - Accuracy, confusion matrix, AUC  
5. **Deployment**  
   - Export model with joblib  
   - Build Streamlit UI  
   - Real-time predictions for uploaded CSV  

---

## 📊 Tech Stack

- **Python**
- **Scikit-learn**
- **Pandas, NumPy**
- **Streamlit**
- **Joblib**

---

##  How to Use the App

1. Prepare a CSV containing **only feature columns** (no diagnosis column).  
2. Upload the file in the Streamlit UI.  
3. View:  
   - **Prediction** (0 = Benign, 1 = Malignant)  
   - **Cancer probability score**  
4. Download the results as CSV.

---

##  Installation (If running locally)

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

##  Developer

**Nikhil Raman**  
AI/ML Engineer | Data Scientist  
https://www.linkedin.com/in/nikhil-raman-k-448589201/

