# -*- coding: utf-8 -*-
"""
LoanPredict - Simple & Elegant Loan Approval Analytics Dashboard
================================================================
Author      : Devangi Yadav
Reg. No.    : 12300822
Roll No.    : 35
Section     : RK2P23FHB35
Institution : Lovely Professional University (LPU)
Dataset     : Loan Approval Classification Dataset (45,000 records)
Model       : Decision Tree Classifier (max_depth=5)
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st

import model_utils
import visualizations as vz

# ----------------------------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="LoanPredict • Devangi Yadav",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, Minimalist Dark Theme Styling (Spacious & Modern)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
    }
    
    /* Global Container Spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1200px !important;
        margin: 0 auto !important;
    }
    
    /* Clean Minimal Header */
    .main-header {
        margin-bottom: 24px;
        padding-bottom: 16px;
        border-bottom: 1px solid #334155;
    }
    .main-title {
        font-size: 26px;
        font-weight: 700;
        color: #f8fafc;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .main-sub {
        font-size: 13.5px;
        color: #94a3b8;
        margin: 0;
    }
    
    /* 4 Spacious KPI Cards */
    .kpi-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    @media (max-width: 768px) {
        .kpi-row { grid-template-columns: repeat(2, 1fr); }
    }
    .kpi-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 18px;
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .kpi-number {
        font-size: 26px;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
    }
    .kpi-note {
        font-size: 11px;
        color: #64748b;
        margin-top: 4px;
    }
    
    /* Clean Cards for Tab Content */
    .clean-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 16px;
    }
    
    /* Decision Alert Boxes */
    .status-badge-approved {
        background: rgba(16, 185, 129, 0.12);
        border: 2px solid #10b981;
        border-radius: 10px;
        padding: 24px;
        text-align: center;
        margin-bottom: 16px;
    }
    .status-badge-rejected {
        background: rgba(244, 63, 94, 0.12);
        border: 2px solid #f43f5e;
        border-radius: 10px;
        padding: 24px;
        text-align: center;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data & Model Loading (Cached)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data...")
def load_data():
    csv_file = "loan_data.csv"
    if not os.path.exists(csv_file):
        raise FileNotFoundError("loan_data.csv not found.")
    df, bad_age_count, income_table = model_utils.load_and_preprocess_data(csv_file)
    return df, bad_age_count, income_table

@st.cache_resource(show_spinner="Training model...")
def load_model(df):
    model, scaler, metrics, splits = model_utils.train_model(df)
    return model, scaler, metrics, splits

try:
    df, bad_age_count, income_table = load_data()
    model, scaler, metrics, splits = load_model(df)
except Exception as e:
    st.error(f"Error loading system: {e}")
    st.stop()


# ----------------------------------------------------------------------------
# Sidebar: Clean Academic Profile
# ----------------------------------------------------------------------------
with st.sidebar:
    if os.path.exists("assets/lpu-logo.jpg"):
        st.image("assets/lpu-logo.jpg", width=160)
        
    st.markdown("### 🎓 Academic Details")
    st.markdown("""
    **Student:** Devangi Yadav  
    **Reg. No:** `12300822`  
    **Roll No:** `35`  
    **Section:** `RK2P23FHB35`  
    **Institution:** Lovely Professional University  
    **Subject:** Machine Learning Practical  
    """)
    st.divider()
    
    st.markdown("### 📌 Quick Summary")
    st.markdown(f"""
    - **Dataset:** 45,000 Applications
    - **Model:** Decision Tree (`depth=5`)
    - **Test Accuracy:** **91.2%**
    - **Data Cleaned:** 7 age outliers fixed
    """)
    
    st.divider()
    if os.path.exists("Predicting_Loan_Approval_Report.docx"):
        with open("Predicting_Loan_Approval_Report.docx", "rb") as f:
            st.download_button(
                "📥 Download Project Report",
                data=f.read(),
                file_name="Predicting_Loan_Approval_Report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )


# ----------------------------------------------------------------------------
# Header Section (Clean & Minimal)
# ----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="main-title">💳 Loan Approval Prediction Dashboard</div>
    <div class="main-sub">
        Decision Tree Classification of 45,000 Loan Applications • <b>Devangi Yadav</b> (Reg: 12300822, Roll: 35, Section: RK2P23FHB35)
    </div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 4 Clean KPI Cards
# ----------------------------------------------------------------------------
overall_approval_pct = float(df["loan_status"].mean() * 100)

st.markdown(f"""
<div class="kpi-row">
    <div class="kpi-box">
        <div class="kpi-label">Total Applications</div>
        <div class="kpi-number">{len(df):,}</div>
        <div class="kpi-note">Full Kaggle dataset</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-label">Overall Approval Rate</div>
        <div class="kpi-number" style="color: #38bdf8;">{overall_approval_pct:.1f}%</div>
        <div class="kpi-note">10,000 approved / 35,000 rejected</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-label">Model Accuracy</div>
        <div class="kpi-number" style="color: #10b981;">{metrics['accuracy']:.1%}</div>
        <div class="kpi-note">Tested on 9,000 unseen applications</div>
    </div>
    <div class="kpi-box">
        <div class="kpi-label">Precision (Approved)</div>
        <div class="kpi-number" style="color: #a855f7;">{metrics['precision']:.1%}</div>
        <div class="kpi-note">1,446 of 1,687 approvals verified</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 4 Simple, Focused Tabs
# ----------------------------------------------------------------------------
tab_overview, tab_model, tab_sim, tab_report = st.tabs([
    "📊 Dataset Overview",
    "🌲 Model & Insights",
    "⚡ Loan Predictor",
    "📄 Report & Downloads"
])


# ============================================================================
# TAB 1: DATASET OVERVIEW
# ============================================================================
with tab_overview:
    col_o1, col_o2 = st.columns(2)
    with col_o1:
        st.plotly_chart(vz.fig_approval_donut(df), use_container_width=True)
    with col_o2:
        st.plotly_chart(vz.fig_income_approval_bars(income_table, overall_rate=overall_approval_pct), use_container_width=True)
        
    st.markdown("""
    <div class="clean-card">
        <h4 style="margin-top:0; color:#f8fafc; font-size:15px;">📌 Key Dataset Insights</h4>
        <ul style="color:#cbd5e1; font-size:13px; margin-bottom:0; line-height: 1.6;">
            <li><b>Imbalanced Data:</b> Only 22.2% of applications are approved. A naive baseline that rejects everyone would have 77.8% accuracy. The Decision Tree reaches <b>91.2%</b>.</li>
            <li><b>Income Group Trend:</b> Approval rates are highest in the Low income group (40.4%) and lowest in the High group (9.3%). In this synthetic dataset, low-income applicants asked for a larger share of their income (18% vs. 10%), which correlated with higher approvals.</li>
            <li><b>Data Cleaning:</b> Blank cells = 0. Exactly <b>7 impossible ages</b> (> 100 years, e.g. 144) were detected and imputed with the median age (26 years).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# TAB 2: MODEL & INSIGHTS
# ============================================================================
with tab_model:
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.plotly_chart(vz.fig_confusion_matrix(metrics), use_container_width=True)
    with col_m2:
        st.plotly_chart(vz.fig_feature_importance(metrics["feature_importances"], top_n=5), use_container_width=True)
        
    st.markdown("""
    <div class="clean-card">
        <h4 style="margin-top:0; color:#f8fafc; font-size:15px;">🔍 What Drives Loan Approval?</h4>
        <ul style="color:#cbd5e1; font-size:13px; margin-bottom:0; line-height: 1.6;">
            <li><b>Previous Default History (47.1% Importance):</b> The single most decisive factor. In this dataset, <b>100% of applicants with a previous default were rejected</b>.</li>
            <li><b>Loan-to-Income Ratio (20.3% Importance):</b> The second most critical branch in the decision tree.</li>
            <li><b>Optimal Depth (max_depth = 5):</b> Training accuracy (91.4%) and test accuracy (91.2%) are balanced, preventing overfitting while keeping decisions transparent.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🌳 View Decision Tree Structure Diagram & Complexity Curve", expanded=False):
        c_tr1, c_tr2 = st.columns([1.2, 1])
        with c_tr1:
            if os.path.exists("outputs/decision_tree.png"):
                st.image("outputs/decision_tree.png", caption="Decision Tree Top 3 Levels (Blue = Approved, Orange = Rejected)", use_container_width=True)
        with c_tr2:
            st.plotly_chart(vz.fig_depth_overfitting(metrics["depth_df"]), use_container_width=True)


# ============================================================================
# TAB 3: LIVE LOAN PREDICTOR
# ============================================================================
with tab_sim:
    st.markdown("##### ⚡ Interactive Loan Approval Simulator")
    st.caption("Adjust the applicant details below to see the instant model prediction and decision rules.")
    
    col_f1, col_f2 = st.columns([1.1, 1])
    
    with col_f1:
        with st.container(border=True):
            r1_1, r1_2 = st.columns(2)
            with r1_1:
                in_age = st.number_input("Age", min_value=18, max_value=85, value=28)
                in_gender = st.selectbox("Gender", ["female", "male"])
                in_edu = st.selectbox("Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"], index=2)
                in_inc = st.number_input("Annual Income ($)", min_value=5000, max_value=500000, value=60000, step=5000)
            with r1_2:
                in_home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"], index=2)
                in_amount = st.number_input("Loan Amount ($)", min_value=500, max_value=50000, value=10000, step=500)
                in_intent = st.selectbox("Loan Purpose", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
                in_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=23.0, value=10.5, step=0.1)
                
            r2_1, r2_2 = st.columns(2)
            with r2_1:
                in_credit = st.slider("Credit Score", min_value=390, max_value=850, value=710, step=5)
            with r2_2:
                in_default = st.selectbox("Previous Default on File?", ["No", "Yes"], index=0)
                
            calc_lpi = in_amount / max(in_inc, 1)
            st.caption(f"Loan-to-Income Ratio: **{calc_lpi:.1%}**")

    with col_f2:
        applicant = {
            "person_age": float(in_age),
            "person_gender": in_gender,
            "person_education": in_edu,
            "person_income": float(in_inc),
            "person_emp_exp": 3,
            "person_home_ownership": in_home,
            "loan_amnt": float(in_amount),
            "loan_intent": in_intent,
            "loan_int_rate": float(in_rate),
            "loan_percent_income": float(calc_lpi),
            "cb_person_cred_hist_length": 4.0,
            "credit_score": int(in_credit),
            "previous_loan_defaults_on_file": in_default
        }
        
        result = model_utils.predict_single_applicant(
            model, scaler, metrics["feature_columns"], applicant
        )
        
        if result["is_approved"]:
            st.markdown(f"""
            <div class="status-badge-approved">
                <h2 style="color: #10b981; margin:0 0 8px 0; font-size:24px;">🎉 LOAN APPROVED</h2>
                <div style="font-size:14px; color:#cbd5e1;">Meets credit underwriting criteria.</div>
                <div style="font-size:18px; font-weight:700; color:#ffffff; margin-top:10px;">
                    Approval Confidence: {result['approval_probability']:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-badge-rejected">
                <h2 style="color: #f43f5e; margin:0 0 8px 0; font-size:24px;">🛑 LOAN REJECTED</h2>
                <div style="font-size:14px; color:#cbd5e1;">Did not meet risk criteria.</div>
                <div style="font-size:18px; font-weight:700; color:#ffffff; margin-top:10px;">
                    Rejection Probability: {result['rejection_probability']:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with st.container(border=True):
            st.markdown("**Underwriting Rule Analysis:**")
            for r in result["reasons"]:
                st.markdown(r)


# ============================================================================
# TAB 4: REPORT & DOWNLOADS
# ============================================================================
with tab_report:
    col_rep1, col_rep2 = st.columns([1.4, 1])
    
    with col_rep1:
        st.markdown(r"""
        #### 📄 Project Practical File Summary
        - **Assigned Topic:** Predicting Loan Approval using Decision Tree Classification
        - **Institution:** Lovely Professional University (LPU)
        - **Submitted By:** Devangi Yadav (Reg. No.: 12300822, Roll No.: 35, Section: RK2P23FHB35)
        
        ##### Problem Statement Tasks Completed:
        1. **Dataset Loaded:** 45,000 past loan applications (14 columns).
        2. **Preprocessing:** 7 impossible ages (> 100) imputed with median (26.0); Credit scores standardized ($\mu=0, \sigma=1$).
        3. **Decision Tree:** Trained with `max_depth=5` to prevent overfitting.
        4. **Evaluation:** Evaluated on 9,000 unseen test rows (80/20 stratified split), reaching **91.2% accuracy**, **85.7% precision**, and **72.3% recall**.
        5. **Income Analysis:** Visualized loan approval dynamics across four income quartiles.
        """)
        
    with col_rep2:
        with st.container(border=True):
            st.markdown("#### 📥 Direct Downloads")
            
            if os.path.exists("Predicting_Loan_Approval_Report.docx"):
                with open("Predicting_Loan_Approval_Report.docx", "rb") as f:
                    st.download_button(
                        "📄 Download Word Report (.docx)",
                        data=f.read(),
                        file_name="Predicting_Loan_Approval_Report.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True
                    )
                    
            if os.path.exists("loan_data.csv"):
                with open("loan_data.csv", "rb") as f:
                    st.download_button(
                        "📊 Download Full Dataset (3.6 MB)",
                        data=f.read(),
                        file_name="loan_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
            if os.path.exists("outputs/results.json"):
                with open("outputs/results.json", "rb") as f:
                    st.download_button(
                        "⚙️ Download Results (.json)",
                        data=f.read(),
                        file_name="results.json",
                        mime="application/json",
                        use_container_width=True
                    )


# ----------------------------------------------------------------------------
# Minimal Footer
# ----------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center; padding: 25px 0 10px 0; color:#64748b; font-size:12px;">
    Lovely Professional University • Devangi Yadav • Registration No: 12300822 • Section: RK2P23FHB35
</div>
""", unsafe_allow_html=True)
