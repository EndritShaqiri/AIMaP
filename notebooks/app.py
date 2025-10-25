import streamlit as st
import numpy as np
import joblib
from lightgbm import LGBMClassifier

# Load trained model
model = joblib.load("models/lightgbm_baseline.pkl")

st.title("🛡️ AIMaP – AI Malware Predictor")
st.write("Upload a PE feature file or drag and drop below to see malware probability.")

uploaded_file = st.file_uploader("Choose a file (.npz or extracted features)")

if uploaded_file is not None:
    # Example: load features (here, mock 2381-length vector)
    X = np.load(uploaded_file)["X"]
    y_pred_proba = model.predict_proba(X)[:,1]
    prob = np.mean(y_pred_proba) * 100

    st.metric("Malicious Probability", f"{prob:.2f}%")

    # Optional: visualize probability distribution
    st.bar_chart(y_pred_proba)
