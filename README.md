# 🛡️ AIMaP – Artificially Intelligent Malware Predictor  

## 📌 Project Description  
**AIMaP** (Artificially Intelligent Malware Predictor) is a machine learning–based malware detection system. Its goal is to analyze Windows Portable Executable (PE) files and:  

- 🔍 Predict the probability that a file is malicious.  
- 🧩 If malicious, classify which **malware family** (e.g., Trojan, Ransomware, Backdoor) it most likely belongs to.  

This project represents the defender’s counterpart to [**AIMaL**](https://github.com/EndritShaqiri/AIMaL) (Artificially Intelligent Malware Launcher). Instead of launching and mutating malware, AIMaP leverages data science to **detect and label malicious files** with high accuracy.  

AIMaP will follow the complete data science lifecycle: **data collection, cleaning, feature extraction, visualization, modeling, evaluation, and deployment** of a lightweight demo interface.  

---

## 🎯 Goals  
- Train ML models to output **malicious probability** for unknown files.  
- Extend classification to **malware families** using multiclass models.  
- Provide **explainability** through feature importance and visualizations.  
- Deliver results in a **clean, reproducible pipeline** hosted on GitHub.  
- Deploy a web demo for file upload, and get
  -  Probability: 92% malicious
  -  Predicted Family: Trojan (confidence 81%)

---

## 📊 Data Collection  
Datasets to be used:  

- **EMBER** → ~2M PE samples with extracted static features for binary classification (malware vs benign).  
- **BODMAS** → 57K malware + 77K benign PE files, labeled by family, with features and metadata.  

Features include:  
- File metadata (size, entropy, virtual size).  
- Imported functions & libraries.  
- Section-level features (names, sizes, entropy).  
- String statistics (count, average length, entropy).  

---

## 🧠 Modeling Approach  

- **Binary Classification → Malware vs Benign**  
  - **Supervised Learning** using LightGBM/XGBoost.  
  - *(Optional extension)* **Unsupervised Learning** using methods such as One-Class SVM, Isolation Forest, or Autoencoders, to detect anomalies without explicit malware labels.  

- **Multiclass Classification → Malware Family Prediction (for malicious samples)**  
  - **Supervised Learning** using LightGBM/XGBoost.  
  - Baselines will include Logistic Regression and Random Forest for comparison.  

- **Evaluation Metrics:**  
  - **Binary:** AUC, Accuracy, Precision, Recall, F1, Confusion Matrix.  
  - **Multiclass:** Accuracy, Macro F1, Confusion Matrix.  

---

## 📈 Data Visualization  
Planned visualizations:  
- Histograms (file size, entropy).  
- Malware family distribution charts.  
- ROC curves for binary classifiers.  
- Confusion matrices for multiclass predictions.  
- Feature importance rankings.  

---

## 🧪 Test Plan  
- Train/test split: **80% / 20%**.  
- Cross-validation + AUC as main metric.  
- Per-family evaluation for multiclass model.   
