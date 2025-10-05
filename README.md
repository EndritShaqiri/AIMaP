# 🛡️ AIMaP – Artificially Intelligent Malware Predictor  

## 📌 Project Description  
**AIMaP** (Artificially Intelligent Malware Predictor) is a machine-learning–based malware detection system designed to analyze Windows Portable Executable (PE) files and:  

- 🔍 Predict the probability that a file is malicious.  
- 🧩 If malicious, classify which **malware family** (e.g., Trojan, Ransomware, Backdoor) it most likely belongs to.  

This project represents the defender’s counterpart to [**AIMaL**](https://github.com/EndritShaqiri/AIMaL).  
Instead of launching and mutating malware, AIMaP leverages data science to **detect and label malicious files** with high accuracy.  
It follows the complete data-science lifecycle: *data collection, cleaning, feature extraction, visualization, modeling, evaluation, and deployment* of a lightweight demo interface.

---

## 🎯 Goals  
- Train ML models to output **malicious probability** for unknown files.  
- Extend classification to **malware families** using multiclass models.  
- Provide **explainability** through feature importance and visualizations.  
- Deploy a web demo for file upload that returns, for example:  
  - Probability: 92% malicious  
  - Predicted Family: Trojan (confidence 81%)  

---

## 📊 Data Collection  

### Datasets  
- **[BODMAS](https://whyisyoung.github.io/BODMAS/)** → 57K malware + 77K benign PE files, labeled by family, with features and metadata.  
- **[EMBER](https://github.com/elastic/ember)** → ~2M PE samples with extracted static features for binary classification (malware vs benign).  

### Planned Usage  
Due to computational constraints, the first phase will focus on the **BODMAS dataset (~130K samples)**.  
If time and resources permit, a second phase will use a **10% subset of EMBER (~200K samples)** — balanced 50% benign / 50% malware — to improve generalization and cross-dataset robustness.  

### Predictor & Target Variables  
- **Predictor variables (features):**  
  - File metadata (size, entropy, virtual size)  
  - Imported functions and libraries  
  - Section-level features (names, sizes, entropies)  
  - String statistics (count, average length, entropy)  

- **Target variables:**  
  - `malicious_label` → 1 = malware, 0 = benign  
  - `malware_family` → e.g., Trojan, Worm, Backdoor, Ransomware  

---

## 🧠 Modeling Approach  

### Binary Classification → Malware vs Benign  
- **Algorithms:** LightGBM, XGBoost  
- *(Optional extension)*: Unsupervised anomaly detection (One-Class SVM, Isolation Forest)  

### Multiclass Classification → Malware Family Prediction  
- **Algorithms:** LightGBM, XGBoost  
- **Baselines:** Logistic Regression, Random Forest  

### Handling Class Imbalance  
- Apply **SMOTE** (Synthetic Minority Oversampling Technique) for underrepresented families.  
- Use **class weighting** in LightGBM/XGBoost to adjust loss contributions.  

### Evaluation Metrics  
- **Binary:** AUC, Accuracy, Precision, Recall, F1, Confusion Matrix  
- **Multiclass:** Accuracy, Macro F1, Confusion Matrix  

---

## 📈 Data Visualization  
Planned visualizations include:  
- Histograms (file size, entropy)  
- Malware-family distribution charts  
- ROC curves for binary classifiers  
- Confusion matrices for multiclass results  
- Feature-importance rankings  

---

## 🧪 Test Plan  
- Dataset partitioning will follow a **chronological split** to better reflect real-world malware evolution:  
  - **Train:** older samples (e.g., 2017 – 2019)  
  - **Test:** newer samples (e.g., 2020 – 2021)  
- Within that split, maintain a standard **80 / 20 ratio**.  
- Apply cross-validation with AUC as the main performance metric.  
- Evaluate per-family metrics for multiclass predictions.  

---

## 🔗 References  
- BODMAS Dataset → [https://whyisyoung.github.io/BODMAS/](https://whyisyoung.github.io/BODMAS/)  
- EMBER Dataset → [https://github.com/elastic/ember](https://github.com/elastic/ember)  
