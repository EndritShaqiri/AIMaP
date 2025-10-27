import streamlit as st
import numpy as np
import joblib
import tempfile
import os
from bodmas_extractor import extract_2381_from_exe

st.set_page_config(page_title="AIMaP - Malware Predictor", page_icon="🛡️")
st.title("🛡️ AIMaP - Malware Predictor (Full 2381 Features)")

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("../models/lightgbm_baseline.pkl")
    scaler = joblib.load("../models/scaler_baseline.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

st.markdown("Upload an `.exe` (requires official extractor).")

uploaded_file = st.file_uploader("Choose a file (.npz, .npy, .exe)", type=["npz","npy","exe"])

if uploaded_file is not None:
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".npz") or name.endswith(".npy"):
            data = np.load(uploaded_file)
            if "X" in data:
                X = data["X"]
                sample = X[0].reshape(1, -1)
            else:
                arr = list(data.values())[0]
                arr = np.array(arr)
                sample = arr.reshape(1, -1) if arr.ndim == 1 else arr

            # No slicing — keep all 2381 features
            if sample.shape[1] != 2381:
                st.error(f"Expected 2381 features, got {sample.shape[1]}")
            else:
                sample_scaled = scaler.transform(sample)
                prob = model.predict_proba(sample_scaled)[0, 1]
                st.success(f"Predicted malware probability: {prob*100:.2f}%")

        elif name.endswith(".exe"):
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".exe")
            tf.write(uploaded_file.read())
            tf.flush()
            tf.close()
            try:
                feats = extract_2381_from_exe(tf.name)
                sample = feats.reshape(1, -1)
                sample_scaled = scaler.transform(sample)
                prob = model.predict_proba(sample_scaled)[0, 1]
                st.success(f"Predicted malware probability: {prob*100:.2f}%")
            finally:
                os.unlink(tf.name)
        else:
            st.error("Unsupported file type.")
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Upload a file to start.")
