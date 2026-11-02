# 💳 LoanPredict • Loan Approval Classification & Analytics Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Predicting Loan Approval using Decision Tree Classification**  
> An end-to-end Machine Learning intelligence dashboard evaluating credit risk, income disparities, and underwriting classification on **45,000 loan applications**.

---

## 🎓 Academic Submission Details
- **Institution:** Lovely Professional University (LPU)
- **Student Name:** Saransh Singh
- **Registration No.:** `12301472`
- **Roll No.:** `35`
- **Section:** `RK2P23FHB36`
- **Report Document:** [`Predicting_Loan_Approval_Report.docx`](Predicting_Loan_Approval_Report.docx)
- **Submission Date:** November 2026

---

## 📊 Key Empirical Results

| Metric | Decision Tree (`max_depth=5`) | Baseline (No-Information Rate) | Improvement |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **91.17%** | 77.78% | **+13.39%** |
| **Precision (Approved)** | **85.71%** | 22.22% | **+63.49%** |
| **Recall (Approved)** | **72.30%** | 100.0% | Balanced capture |
| **F1-Score** | **78.44%** | 36.36% | High harmonic mean |
| **True Negatives (TN)** | **6,759** / 7,000 (96.6% Specificity) | — | Robust fraud protection |
| **False Positives (FP)** | **241** (Wrong approval risk) | — | Kept strictly minimal |

---

## 🚀 Features & Architecture

1. **Executive Cockpit:**
   - Real-time KPI summary (Applications, Approval Rate, Accuracy, Precision, Recall, F1).
   - Approval distribution donut & income quartile analysis.
   - Interactive Confusion Matrix heatmap with loan default explanations.
2. **Exploratory Data Analysis (EDA):**
   - Income vs. Loan Amount scatter visualizer.
   - Categorical breakdown by loan purpose (Personal, Education, Medical, Venture) and home ownership.
   - Credit score distribution before and after standard scaling.
   - Data cleaning audit (median imputation for 7 impossible age values > 100).
3. **Decision Tree Modeling & Complexity:**
   - Feature importance breakdown (Prior Default: 47.1%, Loan-to-Income: 20.3%, Interest Rate: 18.7%).
   - Model depth overfitting analysis (max_depth 1 through 20 proving depth 5 sweet spot).
   - High-resolution visual decision tree architecture rendering.
4. **Live Underwriter Simulator:**
   - Interactive applicant profiling calculator.
   - Real-time decisioning (Approved / Rejected) with confidence score.
   - Triggered risk rules and actionable advisory for applicants.
5. **Report & Artifacts Center:**
   - In-app methodology review.
   - One-click downloads for DOCX report, cleaned CSV dataset, and evaluation metrics JSON.

---

## 📁 Repository Structure

```text
clustering1/
├── app.py                            # Main Streamlit web application
├── model_utils.py                    # Preprocessing, model training & inference engine
├── visualizations.py                 # Interactive Plotly dark-mode chart generators
├── loan_approval_prediction.py       # Standalone batch execution script
├── loan_data.csv                     # 45,000-record dataset
├── Predicting_Loan_Approval_Report.docx # Official project documentation
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git exclusion rules
├── README.md                         # Project documentation
├── assets/
│   └── lpu-logo.jpg                  # University crest / emblem
└── outputs/
    ├── approval_by_income.png        # Bar chart artifact
    ├── confusion_matrix.png          # Heatmap artifact
    ├── dashboard.png                 # Multi-panel composite chart
    ├── decision_tree.png             # Decision tree architecture diagram
    ├── feature_importance.png        # Top feature drivers chart
    └── results.json                  # Serialized evaluation metrics
```

---

## 💻 Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Saransh2005/clustering1.git
   cd clustering1
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit dashboard:**
   ```bash
   streamlit run app.py
   ```
   The dashboard will automatically open at `http://localhost:8501`.

---

## ☁️ How to Deploy on Streamlit Cloud

1. Push this repository to GitHub:
   ```bash
   git add .
   git commit -m "feat: complete loan approval prediction dashboard and report"
   git push -u origin main
   ```
2. Navigate to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in with GitHub (`Saransh2005`).
4. Click **"New app"** and fill in:
   - **Repository:** `Saransh2005/clustering1`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy!"** — Streamlit will automatically install `requirements.txt` and host your live interactive dashboard.

---

## 📖 References
- Kaggle Loan Approval Dataset: [taweilo/loan-approval-classification-data](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)
- Scikit-Learn Documentation: [Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). *Classification and Regression Trees*. Wadsworth.
