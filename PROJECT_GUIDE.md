# ThreatLens AI - Project Specification & Blueprint
**S5 Machine Learning Project Guide**

---

## 1. Project Overview & Objectives

**ThreatLens AI** is an intelligent, Machine Learning-driven Cybersecurity Threat Detection and Analytics platform built for Semester 5 (S5) Computer Science & Machine Learning engineering coursework.

The system processes network flow traffic data from the benchmark **CICIDS2017** dataset, cleans and balances the data, selects high-impact network features, trains three distinct ML models (**Logistic Regression**, **Random Forest**, and **XGBoost**), evaluates their performance, and serves predictions via an interactive **Flask Web Dashboard**.

### Core Objectives
1. **Intelligent Detection**: Accurately classify network traffic as **BENIGN** or specific **ATTACK** types (e.g., DoS/DDoS, Port Scan, Brute Force, Web Attacks).
2. **Confidence Scoring**: Compute probabilistic certainty scores ($0\% - 100\%$) for every prediction.
3. **Severity Classification**: Categorize detected threats into actionable risk levels (**Low**, **Medium**, **High**, **Critical**).
4. **Actionable Recommendations**: Automatically generate targeted security remediation advice for security analysts.
5. **Interactive Security Analytics**: Present live metrics, threat history, and interactive **Plotly** visualizations inside a modern, responsive web dashboard.

---

## 2. System Architecture & 10-Stage Pipeline

```
[ CICIDS2017 Dataset ]
         │
         ▼
[ 1. Data Preprocessing ] ──► (Clean NaNs/Infs, Encode Labels, Standard Scale)
         │
         ▼
[ 2. Imbalance Handling ] ──► (SMOTE / Random Sampling for Minority Attack Classes)
         │
         ▼
[ 3. EDA & Feature Selection ] ──► (Tree Importance & Correlation Selection)
         │
         ▼
[ 4. ML Model Training ] ──► (Logistic Regression, Random Forest, XGBoost)
         │
         ▼
[ 5. Model Comparison ] ──► (Accuracy, F1-Score, Precision, Recall, ROC-AUC)
         │
         ▼
[ 6. Threat Prediction ] ──► (Real-time Inference on Network Flow Samples)
         │
         ▼
[ 7. Confidence Score ] ──► (Calibrated Probability Margin Outputs)
         │
         ▼
[ 8. Severity Classification ] ──► (Low ➔ Medium ➔ High ➔ Critical Risk Rules)
         │
         ▼
[ 9. Security Recommendation ] ──► (Automated Incident Response Action Steps)
         │
         ▼
[ 10. Flask Web Dashboard ] ──► (Interactive Plotly Visuals, SQLite Audit History)
```

---

## 3. Technology Stack

| Technology Layer | Tool / Library | Role & Application |
| :--- | :--- | :--- |
| **Language** | Python 3.12 | Core programming environment |
| **Data Engineering** | Pandas, NumPy, SciPy | DataFrame manipulation, vector calculations, data cleaning |
| **Machine Learning** | Scikit-learn, XGBoost | Preprocessing, feature selection, model training, metrics |
| **Class Imbalance** | Imbalanced-learn (SMOTE) | Synthetic oversampling for rare attack vectors |
| **Model Persistence** | Joblib | Serializing trained models and scalers (`.joblib` / `.pkl`) |
| **Backend & Web API** | Flask, Werkzeug | Lightweight REST web server and routing engine |
| **Database** | SQLite3 | Local storage for prediction history and audit logs |
| **Front-End UI** | HTML5, CSS3, Bootstrap 5 | Modern, responsive, dark-themed dashboard layout |
| **Data Visualization** | JavaScript, Plotly.js / Plotly | Interactive frontend charts, gauge meters, and distributions |

---

## 4. Directory Structure & Repository Layout

```text
ThreatLens-AI/
├── app/
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css            # Custom glassmorphism & dark-mode styling
│   │   └── js/
│   │       ├── dashboard.js         # Interactive Plotly chart rendering
│   │       └── predict.js           # Form validation & AJAX API requests
│   ├── templates/
│   │   ├── base.html                # Shared navbar & UI layout wrapper
│   │   ├── dashboard.html           # Main analytics dashboard & KPI cards
│   │   ├── predict.html             # Threat detection form & sample loader
│   │   ├── analytics.html           # Model comparison & feature importance page
│   │   └── history.html             # SQLite prediction logs table & export
│   ├── app.py                       # Main Flask web server entry point
│   └── database.py                  # SQLite schema setup & query helper functions
├── data/
│   ├── raw/                         # Raw CICIDS2017 CSV dataset files
│   └── processed/                   # Cleaned, balanced & scaled dataset splits
├── models/
│   ├── scaler.joblib                # Fitted StandardScaler artifact
│   ├── feature_names.joblib         # Selected feature names array
│   ├── logistic_regression.joblib   # Trained LR baseline model
│   ├── random_forest.joblib        # Trained Random Forest classifier
│   └── xgboost_model.joblib         # Trained XGBoost classifier
├── notebooks/
│   ├── 01_eda_and_mining.ipynb      # Dataset exploratory visualization
│   └── 02_model_benchmarking.ipynb  # Comparative model experiments
├── results/
│   ├── figures/                     # Confusion matrices, ROC curves, feature importance plots
│   └── metrics/                     # Model performance summaries (JSON / CSV)
├── src/
│   ├── analytics/
│   │   └── feature_selection.py     # Random Forest feature ranking & correlation filter
│   ├── models/
│   │   ├── train.py                 # Multi-model training, tuning & saving pipeline
│   │   └── evaluate.py              # Performance metrics calculation module
│   ├── prediction/
│   │   ├── predictor.py             # Inference loader & confidence scoring engine
│   │   └── recommendation.py        # Severity classification & remediation rules
│   └── preprocessing/
│       ├── cleaning.py              # Missing value, infinity, & label handling
│       └── sampler.py               # SMOTE / random sampling class balancer
├── .gitignore                       # Ignored build & binary artifacts
├── PROJECT_GUIDE.md                 # Complete project specification (this document)
├── README.md                        # High-level repository readme
└── requirements.txt                 # Project dependencies list
```

---

## 5. Detailed Component Specifications

### 5.1 Data Preprocessing & Class Imbalance (`src/preprocessing/`)
- **Cleaning (`cleaning.py`)**:
  - Strips leading/trailing spaces from raw CICIDS2017 column header names.
  - Replaces `Infinity`, `-Infinity`, and missing values (`NaN`) with column medians.
  - Encodes multi-class labels into clean categories (e.g., `BENIGN`, `DoS/DDoS`, `PortScan`, `Brute Force`, `Web Attack`).
- **Class Imbalance Handling (`sampler.py`)**:
  - Raw CICIDS2017 is heavily skewed (~80%+ `BENIGN`).
  - Applies **SMOTE** (Synthetic Minority Over-sampling Technique) or targeted random under-sampling on training splits to ensure rare attack classes are accurately learned by classifiers.

### 5.2 EDA & Feature Selection (`src/analytics/`)
- **Feature Selection (`feature_selection.py`)**:
  - Evaluates feature importance scores using a preliminary **Random Forest Classifier**.
  - Eliminates highly collinear features (correlation $> 0.95$) to prevent multicollinearity.
  - Selects the **top 15-20 most impactful network flow features** (e.g., `Destination Port`, `Flow Duration`, `Total Fwd Packets`, `Packet Length Std`, `Flow Bytes/s`, `Bwd Packet Length Mean`).

### 5.3 Machine Learning Models & Evaluation (`src/models/`)
- **Models Implemented**:
  1. **Logistic Regression**: Linear baseline model for fast binary/multiclass reference.
  2. **Random Forest Classifier**: Non-linear ensemble tree model, robust to outliers.
  3. **XGBoost Classifier**: State-of-the-art gradient boosted decision trees for peak precision and recall.
- **Evaluation Criteria (`evaluate.py`)**:
  - **Accuracy**: Overall classification correctness.
  - **Precision & Recall**: Critical for cybersecurity (minimizing False Positives while capturing all False Negatives).
  - **F1-Score (Macro & Weighted)**: Balanced metric for imbalanced classes.
  - **ROC-AUC & Confusion Matrix**: Visualized per model to identify exact confusion between attack types.

### 5.4 Prediction, Confidence Scoring & Severity Engine (`src/prediction/`)
- **Inference (`predictor.py`)**:
  - Accepts single flow data (via Web form) or batch samples (CSV upload).
  - Scales input features using saved `scaler.joblib`.
  - Runs predictions using the selected model artifact (XGBoost by default).
- **Confidence Scoring**:
  - Extracts prediction probability distribution $P(\text{Class})$.
  - Confidence Score = $\max(P) \times 100\%$.
- **Severity Classification & Recommendations (`recommendation.py`)**:
  - Maps prediction output and confidence level to a threat severity matrix:

| Detected Category | Confidence Level | Severity Level | Actionable Security Recommendation |
| :--- | :--- | :--- | :--- |
| **BENIGN** | Any | **Normal / Low** | Traffic is safe. No defensive action required. |
| **PortScan** | $< 70\%$ | **Low** | Monitor source IP for suspicious scanning patterns. |
| **PortScan** | $\ge 70\%$ | **Medium** | Block scanning source IP at firewall; update rate limits. |
| **DoS / DDoS** | Any | **High / Critical** | Trigger rate-limiting, activate anti-DDoS mitigation, block origin subnet. |
| **Brute Force** | Any | **High** | Enforce multi-factor authentication (MFA); lock targeted user accounts temporarily. |
| **Web Attack** | Any | **Critical** | Inspect Web Application Firewall (WAF) logs; isolate vulnerable web endpoint. |

### 5.5 Database & Prediction History (`app/database.py`)
- SQLite database (`threatlens.db`) maintains a historical audit log table (`predictions`):
  - `id`: Auto-incrementing primary key.
  - `timestamp`: Date and time of prediction.
  - `model_used`: Name of the model used (e.g., XGBoost).
  - `prediction_label`: Predicted class (`BENIGN`, `DoS/DDoS`, etc.).
  - `confidence_score`: Float percentage ($0.0 - 100.0\%$).
  - `severity`: Threat severity level (`Low`, `Medium`, `High`, `Critical`).
  - `feature_summary`: JSON string storing key input feature values.

### 5.6 Interactive Web Dashboard (`app/`)
- Built with **Flask**, **Bootstrap 5**, and **Plotly.js**:
  - **KPI Summary Cards**: Total Scans, Benign Traffic Count, Attacks Detected, High/Critical Alerts.
  - **Real-Time Gauges**: Interactive Plotly gauge showing confidence score of incoming scans.
  - **Analytics Charts**:
    1. Threat Class Distribution (Donut / Pie Chart).
    2. Model Accuracy & F1-Score Benchmarks (Grouped Bar Chart).
    3. Feature Importance Ranking (Horizontal Bar Chart).
    4. Threat Severity Breakdown (Stacked Bar Chart).
  - **Manual Sample Testing**: Input custom network packet metrics or choose preset test samples (e.g., "Simulate DDoS Flow", "Simulate PortScan Flow") to see immediate model predictions.

---

## 6. S5 Scope Boundaries & Guidelines

To ensure the project remains practical, achievable, and strictly aligned with S5 undergraduate ML standards:

### ✅ Included in Scope
- Offline training and testing using the benchmark **CICIDS2017** dataset.
- Standard Machine Learning algorithms (Logistic Regression, Random Forest, XGBoost).
- Class imbalance handling with SMOTE.
- Feature selection and tree-based importance visualization.
- Standard evaluation metrics (Confusion Matrix, Precision, Recall, F1, ROC-AUC).
- Single sample input and CSV batch upload simulation via web UI.
- Local SQLite prediction history storage.
- Interactive Plotly web dashboard served via Flask.

### ❌ Excluded from Current Scope (Future Enhancements)
- **No Deep Learning**: No Neural Networks (ANN/CNN/LSTM) or PyTorch/TensorFlow complexity.
- **No Live Packet Sniffing**: No real-time Scapy/Wireshark network driver integration.
- **No SIEM Integration**: No external enterprise connections (Splunk, Elastic SIEM).
- **No Cloud Infrastructure**: Runs entirely on local machine (`localhost:5000`).
- **No Mobile Applications**: Desktop web interface only.

---

## 7. Recommended Implementation Sequence

```text
Phase 1: Environment Setup & Data Preprocessing
         └── Install dependencies ➔ Load CICIDS2017 ➔ Clean NaNs/Infs ➔ Apply SMOTE

Phase 2: Feature Selection & Model Training
         └── Calculate feature importance ➔ Select Top 15 Features ➔ Train LR, RF & XGBoost

Phase 3: Model Evaluation & Artifact Saving
         └── Export performance metrics & charts ➔ Save scalers and .joblib model files

Phase 4: Prediction & Recommendation Engine
         └── Develop predictor.py ➔ Build confidence & severity classification logic

Phase 5: Flask Application & SQLite Database
         └── Create database schema ➔ Build REST endpoints ➔ Construct Jinja HTML templates

Phase 6: Interactive Plotly Dashboard & Testing
         └── Integrate Plotly.js charts ➔ Verify sample testing form ➔ Final verification
```
