\section*{🛡️ AIMaP – Artificially Intelligent Malware Predictor}

\subsection*{📌 Project Description}
\textbf{AIMaP} (Artificially Intelligent Malware Predictor) is a machine-learning–based malware detection system designed to analyze Windows Portable Executable (PE) files and:
\begin{itemize}
    \item Predict the probability that a file is malicious.
    \item If malicious, classify which \textbf{malware family} (e.g., Trojan, Ransomware, Backdoor) it most likely belongs to.
\end{itemize}

This project represents the defender’s counterpart to \href{https://github.com/EndritShaqiri/AIMaL}{\textbf{AIMaL}} (Artificially Intelligent Malware Launcher). Instead of launching and mutating malware, AIMaP leverages data science to \textbf{detect and label malicious files} with high accuracy. The workflow follows the complete data science lifecycle: \textit{data collection, cleaning, feature extraction, visualization, modeling, evaluation, and deployment} of a lightweight demo interface.

---

\subsection*{🎯 Goals}
\begin{itemize}
    \item Train ML models to output a \textbf{malicious probability} for unknown files.
    \item Extend classification to \textbf{malware families} using multiclass models.
    \item Provide \textbf{explainability} through feature importance and visualization.
    \item Deliver results in a \textbf{clean, reproducible pipeline} hosted on GitHub.
    \item Deploy a web demo for file upload that outputs:
    \begin{itemize}
        \item Probability: 92\% malicious
        \item Predicted Family: Trojan (confidence 81\%)
    \end{itemize}
\end{itemize}

---

\subsection*{📊 Data Collection}
Datasets to be used:
\begin{itemize}
    \item \textbf{BODMAS} – 57K malware + 77K benign PE files, labeled by family, with extracted features and metadata.  
          \href{https://whyisyoung.github.io/BODMAS/}{(whyisyoung.github.io/BODMAS)}
    \item \textbf{EMBER} – $\sim$2M PE samples with extracted static features for binary classification (malware vs. benign).  
          \href{https://github.com/elastic/ember}{(github.com/elastic/ember)}
\end{itemize}

\textbf{Planned usage:}  
Due to computational constraints, the initial training phase will use only the \textbf{BODMAS dataset} (≈130K samples).  
If resources permit, a secondary phase will incorporate about \textbf{10\% of EMBER} (≈200K samples, balanced 50\% benign / 50\% malware) to enhance robustness and cross-dataset generalization.

\textbf{Features (predictor variables):}
\begin{itemize}
    \item File metadata (size, entropy, virtual size)
    \item Imported functions and libraries
    \item Section-level features (names, sizes, entropies)
    \item String statistics (count, average length, entropy)
\end{itemize}

\textbf{Target variables:}
\begin{itemize}
    \item \textbf{Binary classification:} \texttt{malicious\_label} (1 = malware, 0 = benign)
    \item \textbf{Multiclass classification:} \texttt{malware\_family} (e.g., Trojan, Ransomware, Backdoor, Worm)
\end{itemize}

---

\subsection*{🧠 Modeling Approach}
\begin{itemize}
    \item \textbf{Binary Classification (Malware vs. Benign)}  
          Supervised learning using LightGBM and XGBoost.  
          Optional extension: unsupervised anomaly detection using One-Class SVM or Isolation Forest.
    \item \textbf{Multiclass Classification (Malware Family Prediction)}  
          Supervised learning using LightGBM and XGBoost; baselines include Logistic Regression and Random Forest.
\end{itemize}

\textbf{Class imbalance handling:}
\begin{itemize}
    \item Apply \textbf{SMOTE} (Synthetic Minority Oversampling Technique) to balance underrepresented families.
    \item Use \textbf{class weighting} in LightGBM/XGBoost to adjust loss contributions.
    \item Evaluate per-family \textbf{F1-scores} to ensure balanced model performance.
\end{itemize}

\textbf{Evaluation Metrics:}
\begin{itemize}
    \item Binary: AUC, Accuracy, Precision, Recall, F1-score, Confusion Matrix
    \item Multiclass: Accuracy, Macro F1-score, Confusion Matrix
\end{itemize}

---

\subsection*{📈 Data Visualization}
Planned visualizations:
\begin{itemize}
    \item Histograms (file size, entropy)
    \item Malware family distribution charts
    \item ROC curves for binary classifiers
    \item Confusion matrices for multiclass predictions
    \item Feature importance rankings
\end{itemize}

---

\subsection*{🧪 Test Plan}
\begin{itemize}
    \item Dataset partitioning will follow a \textbf{chronological split} to reflect real-world malware evolution:
    \begin{itemize}
        \item Train: older samples (e.g., 2017–2019)
        \item Test: newer samples (e.g., 2020–2021)
    \end{itemize}
    \item Standard 80\% / 20\% division within this temporal split.
    \item Cross-validation with AUC as the main metric.
    \item Per-family evaluation for multiclass models.
\end{itemize}

---

\subsection*{🔗 References}
\begin{itemize}
    \item BODMAS Dataset: \href{https://whyisyoung.github.io/BODMAS/}{https://whyisyoung.github.io/BODMAS/}
    \item EMBER Dataset: \href{https://github.com/elastic/ember}{https://github.com/elastic/ember}
\end{itemize}
