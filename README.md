# 🛡️ AIMaP – Artificially Intelligent Malware Predictor  

🎥 **Midterm Presentation (5 min)** → *[...]*  

---

## 📌 Overview  

**AIMaP (Artificially Intelligent Malware Predictor)** is a machine-learning–based malware detection system designed to analyze Windows Portable Executable (PE) files and predict whether a file is **malicious or benign**.  

The final goal is to build a lightweight, explainable AI-powered antivirus that can take any `.exe` file and output something like:  
> **Probability:** 92% malicious  **Predicted Family:** Trojan (confidence 81%)

---

## 🧩 Work Completed So Far  

### 1️⃣ Data & Features  
- **Dataset:** BODMAS (≈130K PE samples: malware + benign).  
- Each file is represented by **2,381 static features** extracted from PE structure, section info, imports, exports, and string statistics.  
- Working primarily with the official **BODMAS metadata** (feature matrix + labels).  

### 2️⃣ Data Processing  
- Loaded, cleaned, and normalized all features.  
- Applied an **80/20 train-test split** with stratification.  
- Scaled features using `StandardScaler` to standardize distributions.  

### 3️⃣ Modeling – LightGBM Baseline  
A baseline binary classifier was trained to distinguish **malware (1)** vs **benign (0)** files.  

**Model configuration:**
- `n_estimators=300`
- `learning_rate=0.05`
- `num_leaves=64`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `class_weight="balanced"`

**Performance (BODMAS Test Set):**

| Metric | Score |
|---------|-------|
| ✅ Accuracy | **0.9977** |
| ✅ F1-score | **0.9972** |
| ✅ AUC | **0.9999** |

> These near-perfect results confirm that the model captures strong discriminative features between benign and malicious executables.

---

## 💻 Streamlit Web App Prototype  

A functional **Streamlit web demo** (`app.py`) has been developed.  

### ✅ Current Features:
- Upload `.npz` / `.npy` **BODMAS feature files** or raw `.exe` binaries.  
- For `.exe` files, the app automatically extracts the 2,381 features, scales them, and runs prediction through the LightGBM model.  
- Displays the **malware probability** and confidence score.

Example output for a test binary:
> *Predicted malware probability: 78.42%*

---

## 📊 Preliminary Visualizations  

<img width="462" height="391" alt="image" src="https://github.com/user-attachments/assets/d0b92315-c8c7-45ae-9167-d5bc5cd2031a" />
<img width="458" height="391" alt="image" src="https://github.com/user-attachments/assets/90276b89-6ad0-48f5-8d4d-0335de0963e3" />
<img width="789" height="940" alt="image" src="https://github.com/user-attachments/assets/a6b17217-6291-406a-9738-b354818a36b9" />



*(Screenshots and plots are shown in the midterm video presentation.)*

---

## 🚀 Next Steps  

1. **Multiclass Family Classification**  
   - Extend binary classifier to predict malware families (Trojan, Worm, Backdoor, Ransomware).  

2. **Feature Explainability**  
   - Add SHAP analysis to highlight which features contribute most to predictions.  

3. **Cross-Dataset Robustness (EMBER)**  
   - Integrate a 10% subset of the **EMBER dataset (~200K samples)** for additional training and generalization testing.  

4. **Enhanced Streamlit App**  
   - Improve UI for `.exe` uploads, include confidence gauges, visual feature breakdowns, and prediction explanations.

---

## 🧭 Current Takeaways  

- The baseline LightGBM model achieves **>99% accuracy** on BODMAS metadata.  
- Random executable uploads return realistic probability scores (e.g., 2–3% for legit software).  
- Data normalization and stratified splitting were critical for consistent results.  
- The project is modular, well-documented, and ready for expansion to multiclass malware classification and EMBER integration.  

---


---

## ⚙️ How to Run  

```bash
git clone https://github.com/EndritShaqiri/AIMaP
cd AIMaP
cd notebooks
streamlit run app.py


