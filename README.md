# 💳 Credit Risk Scoring Dashboard

A Streamlit-based credit risk scoring application built on a logistic regression pipeline with both the original and SMOTE-balanced datasets included.

---

## 📌 Overview

This repository provides an end-to-end credit default prediction solution:
- `data/raw/Loan_Default.csv`: original borrower dataset
- `data/raw/Loan_Default_balanced.csv`: SMOTE-balanced dataset
- `app/streamlit_app.py`: interactive dashboard for data exploration, model evaluation, and single-borrower scoring
- `balance_data.py`: preprocessing and dataset balancing script
- `tests/test_balance_data.py`: unit tests for the balancing pipeline

The dashboard trains and evaluates a Logistic Regression model using a stratified holdout split and displays interactive metrics and charts.

---

## 📊 Data Summary

### Original dataset
- Total rows: **148,670**
- Columns: **34**
- Target: `Status` (0 = No Default, 1 = Default)
- Class distribution:
  - `0` (No Default): **112,031**
  - `1` (Default): **36,639**
- Default rate: **24.6%**

### Balanced dataset
- File: `data/raw/Loan_Default_balanced.csv`
- Total rows: **224,062**
- Columns: **49** (after one-hot encoding)
- Class distribution:
  - `0` (No Default): **112,031**
  - `1` (Default): **112,031**

The balanced dataset is created using **SMOTE** to address the original class imbalance.

---

## 🧠 Machine Learning Approach

The app uses a **Logistic Regression** pipeline with the following preprocessing steps:
- drop `ID` and unused identifier columns
- median imputation for numeric features
- most-frequent imputation for categorical features
- standard scaling for numeric inputs
- one-hot encoding for categorical inputs
- logistic regression classifier with `class_weight="balanced"`

The dashboard evaluates the model on a **stratified 20% holdout split** and reports standard performance metrics.

---

## 🎯 Dashboard Features

- Executive summary KPI cards
- Default distribution and feature visualizations
- Correlation heatmap and feature importance insights
- Confusion matrix, ROC curve, and classification report
- Real-time borrower risk score prediction
- Option to upload custom CSV data with a `Status` column for retraining

---

## 🗂️ Repository Contents

```
ml-1/
├── app/
│   └── streamlit_app.py          # Interactive Streamlit dashboard
├── balance_data.py               # Preprocessing and SMOTE balancing script
├── data/
│   └── raw/
│       ├── Loan_Default.csv      # Original dataset
│       └── Loan_Default_balanced.csv  # SMOTE-balanced dataset
├── models/
│   ├── logistic_model.pkl        # Saved logistic regression pipeline
│   └── feature_columns.pkl       # Saved feature order for scoring
├── notebooks/
│   └── EDA.ipynb                 # Exploratory analysis notebook
├── tests/
│   └── test_balance_data.py      # Unit tests for data balancing
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧪 Tests

Unit tests are available for the dataset balancing utilities.

```bash
pip install -r requirements.txt
pytest
```

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/RahulThalor/ml-1.git
cd ml-1
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Open **http://localhost:8501** and use the built-in dataset or upload your own CSV with a `Status` column.

---

## 🔧 Regenerate the Balanced Dataset

To refresh the balanced dataset, run:

```bash
python balance_data.py
```

This will recreate `data/raw/Loan_Default_balanced.csv` from the original raw dataset.

---

## 📌 Notes

- `Loan_Default_balanced.csv` is large (~60+ MB). Consider Git LFS if you keep it under source control.
- The dashboard is designed for the schema in `Loan_Default.csv` and requires a `Status` column for supervised scoring.
