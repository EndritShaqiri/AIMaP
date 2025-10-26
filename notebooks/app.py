# app.py (updated)
import streamlit as st
import numpy as np
import joblib
import tempfile
import os

from extract_top20_from_pe import extract_top20_from_path

st.set_page_config(page_title="AIMaP - Malware Predictor", page_icon="🛡️")
st.title("🛡️ AIMaP - Malware Predictor (Top-20 exe demo)")

@st.cache_resource
def load_model():
    return joblib.load("models/lightgbm_top20.pkl")

model = load_model()

st.markdown("Upload a `.npz/.npy` feature file or a raw `.exe`. For `.exe`, the app will extract the Top-20 features and predict.")

uploaded_file = st.file_uploader("Choose a file (.npz, .npy, .exe)", type=["npz","npy","exe"])

if uploaded_file is not None:
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".npz") or name.endswith(".npy"):
            # existing feature vector upload
            data = np.load(uploaded_file)
            if "X" in data:
                X = data["X"]
                sample = X[0].reshape(1, -1)
            else:
                arr = list(data.values())[0]
                arr = np.array(arr)
                if arr.ndim == 1:
                    sample = arr.reshape(1, -1)
                else:
                    sample = arr
            if sample.shape[1] == 2381:
                st.warning("Feature file contains full 2381-dim features. Extracting Top-20 columns from those...")
                top20_indices = [637,2359,2360,2355,658,655,683,613,691,2364,1546,95,2354,626,32,1695,930,255,2375,578]
                sample_top20 = sample[:, top20_indices]
            else:
                # If user uploaded a 20-dim vector directly
                sample_top20 = sample
            prob = model.predict_proba(sample_top20)[0,1]
            st.success(f"Predicted malware probability: {prob*100:.2f}%")

        elif name.endswith(".exe"):
            # save to temp file
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".exe")
            tf.write(uploaded_file.read())
            tf.flush()
            tf.close()
            try:
                feats = extract_top20_from_path(tf.name)  # returns 20-dim np.array
                sample = feats.reshape(1, -1)
                prob = model.predict_proba(sample)[0,1]
                st.success(f"Predicted malware probability: {prob*100:.2f}%")
                st.write("Top-20 feature vector:")
                st.write(sample[0][:20].tolist())
            finally:
                os.unlink(tf.name)
        else:
            st.error("Unsupported file type.")
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Upload a file to start.")
