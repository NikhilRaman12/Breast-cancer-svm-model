#import all the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, RocCurveDisplay
import joblib
import pickle
 #step 1: Load the dataset
data= pd.read_csv("breast-cancer.csv")
print(data.head())
#step 2: Eda and Preprocessing
print(data.shape)
print(data.info)
print(data.isnull().sum())
print(data.describe())
print(data.dtypes)
print(data.columns)
#step3
#Visualization
sns.histplot(data["diagnosis"],kde=True)
plt.show()
#correlation analysis
numeric_data= data.select_dtypes(include=["number"])
corr=numeric_data.corr( method= 'pearson')
plt.show()
sns.heatmap(corr, cmap="Blues", annot=True)
plt.show()
#step 4: feature engineering and data transformations
data["radius_growth"] = data["radius_worst"]- data["radius_mean"]
data["mean_area_perimeter_ratio"]= data["area_mean"]/ data["perimeter_mean"]
data["worst_area_perimeter_ratio"]= data["area_worst"]/ data["perimeter_worst"]
data["worst_radius_texture_ratio"]= data["radius_worst"]/data["texture_mean"]
data["compactness_worst_ratio"]= data["compactness_mean"]/data["compactness_worst"]
data["concavity_worst_ratio"]= data["concavity_mean"]/data["concavity_worst"]
#Encode the diagnosis
data["diagnosis"]= data["diagnosis"].map({"M": 1, "B":0})
#Texture Morphology Interaction Features (Custom Nonlinear Features)
data["texture_area_interaction"]= data["texture_mean"]*data["area_mean"]
data["smoothness_perimeter"]= data["smoothness_mean"]*data["perimeter_mean"]
data["radius_texture"]=data["radius_mean"]*data["texture_mean"]
data["symmetry_compactness"]= data["symmetry_mean"]*data["compactness_mean"]

#polynomial features
data["radius_mean_squared"]= data["radius_mean"]**2
data["area_mean_squared"]= data["area_mean"]**2
data["smoothness_mean_squared"]= data["smoothness_mean"]**2
data["compactness_mean_squared"]= data["compactness_mean"]**2
data["concavity_mean_squared"]= data["concavity_mean"]**2
#log transform for skewed features
skewed=["area_mean", 'area_worst', 'concavity_mean', 'concave points_mean']
for col in skewed:
    data[col+ "_log"]= np.log1p(data[col])

# Replace inf values with NaN after feature engineering
data.replace([np.inf, -np.inf], np.nan, inplace=True);

# Drop rows with NaN values before splitting and scaling
data.dropna(inplace=True)

#step 5: define  features and target(X and y)]
X= data.drop("diagnosis", axis=1)
y= data["diagnosis"]
#step 6: define model and train
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler= StandardScaler()
X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled= scaler.transform(X_test)
SVC_model= SVC(kernel= "rbf", gamma= "scale",   C=10, probability= True,
               random_state=42)
SVC_model.fit(X_train_scaled, y_train)
#step 7: pedict the binary outputs(0,1)
y_pred= SVC_model.predict(X_test_scaled)
#for ROC-AUC
y_prob= SVC_model.predict_proba(X_test_scaled)[:, 1]
#step 8 Evaluate the model using metrics
print("Accuracy", accuracy_score(y_test, y_pred))
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Greens", fmt='d')
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", roc_auc)

RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve")
plt.show()

#save the model and scaler
joblib.dump(SVC_model, "svm_breast_cancer_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")
# Also create a blank CSV template with correct headers
pd.DataFrame(columns=X.columns).to_csv("upload_template.csv", index=False)
X.iloc[:10].to_csv("upload_template.csv", index=False)
