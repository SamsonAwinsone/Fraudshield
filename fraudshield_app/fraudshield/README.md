# 🛡️ FraudShield — Credit Card Fraud Detection Platform

**OMIS 304: Machine Learning 1 | Group 5 | University of Ghana | 2023**

A multi-page, production-quality Streamlit application that classifies credit card
transactions as **Fraud** or **Legitimate** using SVM, Random Forest, and XGBoost,
with SMOTE oversampling and an animated real-time risk gauge.

---

## 👥 Team Members

| Name | Student ID | Contribution |
|------|-----------|--------------|
| Borketey Pearl Borley | 11015547 | Data Preprocessing & SMOTE |
| Brown Shizzel | 11195549 | SVM Model & Calibration |
| Yann Yobeu | 11004336 | Random Forest & Feature Importance |
| Brandford Mettle | 11259497 | XGBoost Modelling & ROC Analysis |
| Nketsiah Obed | 11296843 | Streamlit UI/UX & Prediction Interface |

---

## 📁 Project Structure

```
fraudshield/
├── app.py                  ← Main Streamlit application (all 7 pages)
├── fraudTest.csv           ← Dataset (from Kaggle, kartik2112/fraud-detection)
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
└── .streamlit/
    └── config.toml         ← Streamlit theme configuration
```

---

## 📊 Dataset

- **Source:** https://www.kaggle.com/datasets/kartik2112/fraud-detection
- **File:** `fraudTest.csv`
- **Size:** 555,719 transactions
- **Fraud Rate:** 0.39% (severe class imbalance)
- **Features:** 22 columns including transaction time, amount, merchant category,
  cardholder demographics, and geographic coordinates

---

## 🔧 Feature Engineering (10 Features)

| Feature | Source | Rationale |
|---------|--------|-----------|
| `log_amt` | `amt` | Log-normalises right-skewed transaction amounts |
| `hour` | `trans_date_trans_time` | Fraud peaks at night (00:00–03:00) |
| `day_of_week` | `trans_date_trans_time` | Weekend fraud is elevated |
| `month` | `trans_date_trans_time` | Seasonal fraud patterns |
| `is_weekend` | `day_of_week` | Binary flag for weekend |
| `age` | `dob` | Cardholder age (fraud susceptibility proxy) |
| `city_pop` | direct | Urban-density socioeconomic proxy |
| `geo_distance` | `lat/long` + `merch_lat/long` | Haversine distance (km) — stolen card signal |
| `gender_enc` | `gender` | F=0, M=1 |
| `category_enc` | `category` | Label-encoded merchant category |

---

## 🤖 Models

| Model | Key Config | Notes |
|-------|-----------|-------|
| **SVM** | LinearSVC + Platt Calibration | O(n) scalable; calibrated probabilities |
| **Random Forest** | 200 trees, `class_weight='balanced'` | Interpretable feature importances |
| **XGBoost** ★ Bonus | 200 trees, `learning_rate=0.1` | State-of-the-art tabular performance |

**Imbalance handling:** SMOTE oversampling applied to **training set only**
(never the test set — avoids inflated metrics).

**Validation:** Stratified holdout split (80% train / 20% test).

---

## 🚀 Local Setup & Run

### 1. Clone / Download the project
```bash
git clone https://github.com/YOUR_USERNAME/fraudshield.git
cd fraudshield
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Place `fraudTest.csv` in the project root directory.
Download from: https://www.kaggle.com/datasets/kartik2112/fraud-detection

### 5. Run the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## ☁️ Deployment on Streamlit Community Cloud

### Step 1 — Push to GitHub
```bash
git init
git add app.py requirements.txt README.md .streamlit/config.toml
# Note: fraudTest.csv is large (80MB). Use Git LFS or the upload feature.
git commit -m "Initial FraudShield deployment"
git remote add origin https://github.com/YOUR_USERNAME/fraudshield.git
git push -u origin main
```

### Step 2 — Handle the large CSV
Option A — Git LFS (recommended):
```bash
git lfs install
git lfs track "*.csv"
git add .gitattributes fraudTest.csv
git commit -m "Add dataset via LFS"
git push
```

Option B — Sidebar upload:
- Do NOT commit the CSV
- The app's sidebar has a file uploader fallback
- Users upload `fraudTest.csv` at runtime

### Step 3 — Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io
2. Click **New app**
3. Connect your GitHub repo
4. Set **Main file path:** `app.py`
5. Click **Deploy**

Your app gets a public URL:
`https://YOUR_USERNAME-fraudshield-app-HASH.streamlit.app`

---

## 📈 Expected Model Performance (40K sample, seed=42)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| SVM | ~81% | ~1–5% | ~65–75% | ~3–9% | ~83% |
| Random Forest | ~99.4% | ~25–35% | ~35–45% | ~30–40% | ~96% |
| XGBoost ★ | ~99.7% | ~55–65% | ~50–60% | ~55–60% | ~97–98% |

> 💡 **Tip:** Increase sample size to 60,000–80,000 for more fraud samples in
> the test set and more stable metrics. Training time: ~2–4 min at 80K.

---

## 🗺️ App Pages

| # | Page | Description |
|---|------|-------------|
| 1 | 🏠 Home & Dashboard | KPI strip, fraud charts, business insight |
| 2 | 📊 Data Explorer | Raw data, statistics, distributions, correlation |
| 3 | ⚙️ Preprocessing | Feature engineering, SMOTE before/after |
| 4 | 🤖 Model Training | Config, train all 3 models, feature importances |
| 5 | 📈 Evaluation | Metrics, radar, confusion matrices, ROC curves |
| 6 | 🔮 Fraud Predictor | Live form → animated risk gauge → decision |
| 7 | 💡 Insights | Feature insights, model trade-offs, recommendations |

---

## 📋 Requirements

```
streamlit>=1.32.0
pandas>=1.5.0
numpy>=1.23.0
plotly>=5.18.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
xgboost>=1.7.0
```

---

## ⚠️ Academic Integrity

This project is submitted for OMIS 304: Machine Learning 1,
University of Ghana, Department of Operations and Management Information Systems.
All rights reserved © Group 5, 2023.
