# AIMaP — Artificially Intelligent Malware Predictor

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-aim--sec.com-blue?style=for-the-badge&logo=github)](https://aim-sec.com/)

AIMaP is an AI-powered malware analysis engine designed to deliver high-accuracy malicious-file detection and malware family classification. It analyzes PE files (EXE, DLL, SYS), PDFs, and ELF binaries (with highest accuracy on PE files) using static features such as entropy, imports, section metadata, structural patterns, and authenticated signatures.

When a file is uploaded, AIMaP extracts static features, computes cryptographic hashes, and generates a probabilistic assessment of maliciousness. If the file is malicious, AIMaP additionally predicts the most likely malware family.

AIMaP's machine-learning models were trained on over 5 million real-world samples collected from VirusTotal (2023–2024) using the EMBER2024 dataset, covering thousands of malware families and a large distribution of benign software.

## 🎯 Live Demo

** [Try AIMaP Now: https://aim-sec.com/](https://aim-sec.com/)**

## Dataset Statistics

| File Type | Malicious + Benign (Weekly) | Train Total | Test Total |
|-----------|-----------------------------|-------------|------------|
| Win32     | 30,000                      | 3,120,000   | 720,000    |
| Win64     | 10,000                      | 1,040,000   | 240,000    |
| .NET      | 5,000                       | 520,000     | 120,000    |
| APK       | 4,000                       | 516,000     | 96,000     |
| PDF       | 1,000                       | 104,000     | 24,000     |
| ELF       | 500                         | 52,000      | 12,000     |

- **Total training set:** 5,252,000 files
- **Total test set:** 1,212,000 files
- **Dataset size:** ~50 GB

Samples were collected daily from September 24th 2023 to December 14th 2024. This ensures fresh, modern malware and reduces dataset staleness.

To remove near-duplicate files, AIMaP used Trend Micro TLSH (Locality Sensitive Hashing). Any file whose TLSH distance was below 30 from an existing file was removed.

## Data Vectorization & Splitting

Features are vectorized, stored, and normalized (`model.py`, `binary_vectorizer.py`, `family_vectorizer.ipynb`).

Instead of a typical 80/20 split, AIMaP uses a chronological split across 64 weeks:

- **Weeks 1–52 → training** (older samples)
- **Weeks 53–64 → testing** (newer samples)

Family labels are assigned using ClarAVy (used by EMBER2024), which applies Bayesian inference and provides both a family label and a confidence score.

All features for the neural network are standardized with StandardScaler.

## Feature Extraction

AIMaP uses a static-analysis PE feature extractor (`PEFeatureExtractor` in `features.py`) that converts each binary into a fixed-length vector of **2,568 features**.

Extracted feature groups include:

- **General File Information:** file size, entropy, valid PE flag, first 4 bytes
- **Byte Histogram (256-dim):** useful for detecting packing and encryption
- **Byte-Entropy Histogram:** joint byte/entropy patterns across sliding windows
- **String Features:** URLs, IPs, registry keys, PowerShell commands, download/connect indicators, etc.
- **Section Information:** raw size, virtual size, entropy, RWX flags, overlay stats
- **Import Features:** DLL names and imported APIs (hashed to fixed dimension)
- **Authenticode Signature Features:** certificate count, validity, signer data
- and much more.

These features collectively capture structural, statistical, and semantic file characteristics.

## Models

AIMaP uses two separate machine-learning models:

1. **LightGBM** for binary malware detection
2. **Deep Neural Network (PyTorch MLP)** for malware family classification

Both models and their configuration are documented in:  
`/notebooks/aimap.ipynb`

## Performance

### Binary Classifier
- **Test AUC:** 0.9981395167249398
- **Average Precision:** 0.9982659016220766
- Confusion matrix:

 ![Binary Classifier Confusion Matrix](/pics/binary.png)

### Family Classifier
From 2,358 malware families, the top 50 families were selected for training.
- **Train Accuracy:** 0.9359
- **Validation Accuracy:** 0.9277
- **Test Accuracy:** 0.8561
- Confusion matrix:

 <img src="/pics/family.png" width="75%" alt="Family Classifier Confusion Matrix">

## Inference Pipeline

The file `aimap_core.py` implements the entire inference pipeline, including:

1. Loading all pre-trained models
2. Extracting features from uploaded PE files
3. Predicting maliciousness
4. Predicting malware family (only if malicious)

This module powers the backend used by the web interface.

## Frontend

All frontend code is located in `/web/`.  
The JavaScript files handle uploading, API communication, and displaying model predictions.

## Future Work

While the binary classifier already achieves excellent results, future work will focus on:

- Improving family-level classification
- Expanding into behavior-type prediction
