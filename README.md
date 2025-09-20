# AIMaP
🛡️ AIMaP – Artificially Intelligent Malware Predictor
📌 Project Description

AIMaP (Artificially Intelligent Malware Predictor) is a machine learning–based malware detection system. Its goal is to analyze Windows Portable Executable (PE) files and:

Predict the probability that a file is malicious.

If malicious, classify which malware family (e.g., Trojan, Ransomware, Backdoor) it most likely belongs to.

This project represents the defender’s counterpart to AIMaL (Artificially Intelligent Malware Launcher). Instead of launching and mutating malware, AIMaP leverages data science to detect and label malicious files with high accuracy.

AIMaP will follow the complete data science lifecycle: data collection, cleaning, feature extraction, visualization, modeling, evaluation, and deployment of a lightweight demo interface.

🎯 Project Goals

Train ML models to output malicious probability for unknown files.

Extend classification to malware families using multiclass models.

Provide explainability through feature importance and visualizations.

Deliver results in a clean, reproducible pipeline hosted on GitHub.

(Optional) Deploy a simple one-page web interface for file prediction.

📊 Data Collection

We will use two major malware datasets:

EMBER 2018 → ~1.1M PE samples with extracted static features for binary classification (malware vs benign).

BODMAS → 57K malware + 77K benign PE files, labeled by family, with features and metadata.

These datasets include:

File metadata (size, entropy, virtual size).

Imported functions & libraries.

Section-level features (names, sizes, entropy).

String statistics (count, average length, entropy).

🧠 Modeling Approach

Binary classification (Malware vs Benign) using LightGBM/XGBoost.

Multiclass classification (Malware Family) for malicious samples.

Baseline models (Logistic Regression, Random Forest) will be tested first, followed by advanced gradient-boosting methods.

Evaluation Metrics:

Binary: AUC, Accuracy, Precision, Recall, F1, Confusion Matrix.

Multiclass: Accuracy, Macro F1, Confusion Matrix.

📈 Data Visualization

We will use visualizations to compare malware vs benign distributions and family-level patterns:

Histograms (file size, entropy).

Malware family distribution charts.

ROC curves for binary models.

Confusion matrices for multiclass results.

Feature importance rankings.

🧪 Test Plan

Split datasets into 80% training / 20% testing.

Validate models using cross-validation and AUC scores.

Report per-family metrics for classification.

Compare performance across EMBER and BODMAS subsets.

🗓️ Project Timeline

Week 1–2: Dataset collection, exploration, cleaning.

Week 3–4: Baseline models for binary detection.

Week 5–6: LightGBM/XGBoost binary classifier.

Week 7: Extend to malware family classification.

Week 8–9: Visualizations, feature importance, testing.

Week 10: Final polish, documentation, presentation prep.

⚙️ Deliverables

GitHub repo with:

README.md (proposal, midterm, final report).

Scripts/notebooks for data prep, modeling, and evaluation.

Visualizations (ROC, confusion matrices, feature importance).

Midterm + Final presentation videos.

(Optional) Streamlit web demo (localhost) to upload a file and get:
Probability: 92% malicious  
Predicted Family: Trojan (confidence 81%)  
