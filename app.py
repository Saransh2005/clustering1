# -*- coding: utf-8 -*-
"""
LoanPredict - Executive Loan Approval Classification Dashboard
==============================================================
Author      : Saransh Singh
Reg. No.    : 12301472
Roll No.    : 35
Section     : RK2P23FHB36
Institution : Lovely Professional University (LPU)
Dataset     : Loan Approval Classification Dataset (45,000 records)
Model       : Scikit-Learn DecisionTreeClassifier (max_depth=5)
"""

import os
import json
import base64
import pandas as pd
import numpy as np
import streamlit as st

import model_utils
import visualizations as vz

# ----------------------------------------------------------------------------
# Page Config & Theming
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="LoanPredict • Decision Tree Analytics",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Fintech Cockpit)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
    }
    
    /* Top Bar adjustments */
    header[data-testid="stHeader"] {
        background: rgba(15, 23, 42, 0.85) !important;
        backdrop-filter: blur(8px);
    }
    
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
        max-width: 100% !important;
    }
    
    /* Hero Header */
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 18px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        font-size: 26px;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.5px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 5px;
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
    }
    .hero-badge {
        font-size: 11px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .hero-badge-green {
        background: rgba(16, 185, 129, 0.12);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .hero-badge-purple {
        background: rgba(168, 85, 247, 0.12);
        color: #a855f7;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    
    /* KPI Metric Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 12px;
        margin-bottom: 18px;
    }
    .kpi-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94a3b8;
    }
    .kpi-val {
        font-size: 24px;
        font-weight: 800;
        color: #f8fafc;
        margin: 4px 0 2px 0;
    }
    .kpi-sub {
        font-size: 11px;
        color: #64748b;
    }
    
    /* Content Card Container */
    .panel-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
    }
    
    /* Predictor Result Cards */
    .predict-approved {
        background: rgba(16, 185, 129, 0.12);
        border: 2px solid #10b981;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .predict-rejected {
        background: rgba(244, 63, 94, 0.12);
        border: 2px solid #f43f5e;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data & Model Caching
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and preprocessing dataset...")
def get_dataset():
    # If loan_data.csv exists, use it
    csv_file = "loan_data.csv"
    if not os.path.exists(csv_file):
        raise FileNotFoundError("loan_data.csv was not found in directory.")
    df, bad_age_count, income_table = model_utils.load_and_preprocess_data(csv_file)
    return df, bad_age_count, income_table

@st.cache_resource(show_spinner="Training Decision Tree Model...")
def get_trained_model(df):
    model, scaler, metrics, splits = model_utils.train_model(df)
    return model, scaler, metrics, splits

try:
    df_full, bad_age_count, income_table = get_dataset()
    model, scaler, metrics, splits = get_trained_model(df_full)
except Exception as e:
    st.error(f"Error initializing data or model: {e}")
    st.stop()


# ----------------------------------------------------------------------------
# Sidebar Configuration & Filters
# ----------------------------------------------------------------------------
with st.sidebar:
    # Logo / Affiliation
    if os.path.exists("assets/lpu-logo.jpg"):
        st.image("assets/lpu-logo.jpg", width=180)
        
    st.markdown("### 🎓 Academic Project Details")
    st.markdown("""
    **Student:** Saransh Singh  
    **Reg. No:** `12301472`  
    **Roll No:** `35`  
    **Section:** `RK2P23FHB36`  
    **Course:** Machine Learning Practical  
    **Date:** November 2026  
    """)
    st.divider()
    
    st.markdown("### 🎛️ Interactive Filters")
    st.caption("Filter records across exploratory visualizations:")
    
    # Home ownership filter
    home_options = list(df_full["person_home_ownership"].unique())
    selected_home = st.multiselect("Home Ownership", home_options, default=home_options)
    
    # Loan intent filter
    intent_options = list(df_full["loan_intent"].unique())
    selected_intent = st.multiselect("Loan Purpose / Intent", intent_options, default=intent_options)
    
    # Previous default filter
    default_filter = st.radio("Prior Default History", ["All Applications", "Clean Credit Only (No Default)", "Previous Default on File (Yes)"])
    
    # Income range filter
    min_inc = int(df_full["person_income"].min())
    max_inc = min(int(df_full["person_income"].quantile(0.99)), 250000)
    income_slider = st.slider("Annual Income Range ($)", min_value=min_inc, max_value=max_inc, value=(min_inc, max_inc), step=5000)
    
    # Apply filtering
    mask = (
        df_full["person_home_ownership"].isin(selected_home) &
        df_full["loan_intent"].isin(selected_intent) &
        (df_full["person_income"] >= income_slider[0]) &
        (df_full["person_income"] <= income_slider[1])
    )
    if default_filter == "Clean Credit Only (No Default)":
        mask = mask & (df_full["previous_loan_defaults_on_file"] == "No")
    elif default_filter == "Previous Default on File (Yes)":
        mask = mask & (df_full["previous_loan_defaults_on_file"] == "Yes")
        
    filtered_df = df_full[mask].copy()
    
    st.divider()
    st.metric("Filtered Records", f"{len(filtered_df):,}", f"{(len(filtered_df)/len(df_full))*100:.1f}% of total")
    if st.button("Reset Filters", use_container_width=True):
        st.rerun()


# ----------------------------------------------------------------------------
# Hero Banner
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero-card">
    <div class="hero-title">
        <span>💳</span> Predicting Loan Approval • Decision Tree Analytics
    </div>
    <div class="hero-subtitle">
        An enterprise end-to-end Machine Learning intelligence dashboard evaluating credit risk, income disparities, and loan approval classification.
    </div>
    <div class="hero-badges">
        <span class="hero-badge">Lovely Professional University</span>
        <span class="hero-badge">Saransh Singh • 12301472</span>
        <span class="hero-badge-green">Test Accuracy: 91.17%</span>
        <span class="hero-badge-purple">45,000 Loan Records</span>
        <span class="hero-badge">DecisionTree max_depth=5</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Executive KPI Metric Grid
# ----------------------------------------------------------------------------
overall_appr_rate = (df_full["loan_status"].mean()) * 100
filtered_appr_rate = (filtered_df["loan_status"].mean() * 100) if len(filtered_df) > 0 else 0.0

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Dataset Applications</div>
        <div class="kpi-val">{len(df_full):,}</div>
        <div class="kpi-sub">Filtered: {len(filtered_df):,} rows</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Overall Approval Rate</div>
        <div class="kpi-val" style="color: #38bdf8;">{overall_appr_rate:.1f}%</div>
        <div class="kpi-sub">Filtered rate: {filtered_appr_rate:.1f}%</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Model Test Accuracy</div>
        <div class="kpi-val" style="color: #10b981;">{metrics['accuracy']:.2%}</div>
        <div class="kpi-sub">Baseline: 77.78% (+13.4%)</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Precision (Approved)</div>
        <div class="kpi-val" style="color: #10b981;">{metrics['precision']:.1%}</div>
        <div class="kpi-sub">1,446 / 1,687 true hits</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Recall (Approved)</div>
        <div class="kpi-val" style="color: #f59e0b;">{metrics['recall']:.1%}</div>
        <div class="kpi-sub">1,446 / 2,000 captured</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">F1-Score Balance</div>
        <div class="kpi-val" style="color: #a855f7;">{metrics['f1']:.1%}</div>
        <div class="kpi-sub">Harmonic Mean</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Main Navigation Tabs
# ----------------------------------------------------------------------------
tab_cockpit, tab_eda, tab_model, tab_simulator, tab_report = st.tabs([
    "📊 Executive Cockpit",
    "🔍 Exploratory Analysis (EDA)",
    "🌲 Decision Tree Model & Evaluation",
    "⚡ Real-Time Underwriter Simulator",
    "📄 Project Report & Artifacts"
])


# ============================================================================
# TAB 1: EXECUTIVE COCKPIT
# ============================================================================
with tab_cockpit:
    st.markdown("### 📊 High-Level Decision Analytics")
    
    col_c1, col_c2 = st.columns([1, 1.3])
    with col_c1:
        st.plotly_chart(vz.fig_approval_donut(filtered_df), use_container_width=True)
    with col_c2:
        st.plotly_chart(vz.fig_income_approval_bars(income_table, overall_rate=overall_appr_rate), use_container_width=True)
        
    st.markdown("""
    > 💡 **Key Dataset Paradox (Income vs. Approval Rate):**  
    > In this synthetic dataset, **approval rate falls as annual income rises** (40.4% in the Low group down to 9.3% in the High group).  
    > **Root Cause:** Low-income applicants request a significantly higher loan-to-income proportion (average 18% vs. 10% for high-income), and in this synthetic data generation process, loan-to-income ratio correlates positively with approval status.
    """)
    
    st.divider()
    
    col_c3, col_c4 = st.columns([1.1, 1])
    with col_c3:
        st.plotly_chart(vz.fig_confusion_matrix(metrics), use_container_width=True)
    with col_c4:
        st.plotly_chart(vz.fig_feature_importance(metrics["feature_importances"], top_n=6), use_container_width=True)


# ============================================================================
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
with tab_eda:
    st.markdown("### 🔍 Exploratory Data Analysis & Anomaly Auditing")
    
    # Missing values and cleaning callout
    with st.expander("🛠️ Data Cleaning & Anomaly Auditing Summary", expanded=False):
        st.markdown(rf"""
        - **Total Dataset Size:** 45,000 applications, 14 original features.
        - **Missing Blank Cells in File:** Exactly 0 empty cells.
        - **Anomaly Detection (Age > 100):** Identified **7 impossible age entries** (ranging from 109 to 144 years).
        - **Imputation Strategy:** Replaced bad age entries with the dataset median age (**{df_full['person_age'].median():.0f} years**).
        - **Credit Score Standardization:** Scaled from $\mu = {metrics['cs_mean_raw']:.2f}, \sigma = {metrics['cs_std_raw']:.2f}$ to $\mu = 0.00, \sigma = 1.00$ on train split before testing.
        """)
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.plotly_chart(vz.fig_scatter_income_loan(filtered_df), use_container_width=True)
    with col_e2:
        st.plotly_chart(vz.fig_credit_score_standardization(df_full["credit_score"]), use_container_width=True)
        
    col_e3, col_e4 = st.columns(2)
    with col_e3:
        st.plotly_chart(vz.fig_categorical_approval(filtered_df, "loan_intent", "Approval Rate by Loan Purpose / Intent"), use_container_width=True)
    with col_e4:
        st.plotly_chart(vz.fig_categorical_approval(filtered_df, "person_home_ownership", "Approval Rate by Home Ownership"), use_container_width=True)
        
    st.markdown("#### 📋 Filtered Records Explorer")
    st.dataframe(
        filtered_df.head(100),
        use_container_width=True,
        hide_index=True
    )


# ============================================================================
# TAB 3: MODEL PERFORMANCE & DECISION TREE
# ============================================================================
with tab_model:
    st.markdown("### 🌲 Decision Tree Classification Architecture")
    
    col_m1, col_m2 = st.columns([1.1, 1])
    with col_m1:
        st.plotly_chart(vz.fig_depth_overfitting(metrics["depth_df"]), use_container_width=True)
    with col_m2:
        st.plotly_chart(vz.fig_feature_importance(metrics["feature_importances"], top_n=8), use_container_width=True)
        
    st.markdown("""
    #### 🎯 Why Maximum Depth 5 is Optimal:
    - **Overfitting Avoidance:** At `max_depth=5`, Training Accuracy (**91.4%**) and Test Accuracy (**91.17%**) are virtually identical, showing strong generalizability.
    - **Higher Depths Memorize:** If `max_depth` is allowed to grow to 20, Training Accuracy jumps to **98.43%**, but Test Accuracy deteriorates to **90.28%** (pure overfitting).
    - **Model Explainability:** Depth 5 limits decision paths to at most 5 sequential questions, complying with financial regulatory transparency requirements.
    """)
    
    st.divider()
    
    st.markdown("#### 🌳 Trained Decision Tree Visualization (Top Levels)")
    if os.path.exists("outputs/decision_tree.png"):
        st.image("outputs/decision_tree.png", caption="Decision Tree Structure (Blue = Approved, Orange = Rejected, max_depth=3 view)", use_container_width=True)
    elif os.path.exists("extracted_media/25ef8fa6ea7ae012fdaf13e63b03cd5cf3ac14bf.png"):
        st.image("extracted_media/25ef8fa6ea7ae012fdaf13e63b03cd5cf3ac14bf.png", caption="Decision Tree Structure", use_container_width=True)
    else:
        st.info("Decision tree visual file generated in outputs folder.")


# ============================================================================
# TAB 4: REAL-TIME UNDERWRITING SIMULATOR
# ============================================================================
with tab_simulator:
    st.markdown("### ⚡ Live Loan Underwriter Simulator")
    st.caption("Enter prospective applicant details to generate immediate real-time decisioning and risk path explanation.")
    
    col_sim_in, col_sim_out = st.columns([1.2, 1])
    
    with col_sim_in:
        st.markdown("##### 📝 Applicant Profile")
        
        c_in1, c_in2 = st.columns(2)
        with c_in1:
            in_age = st.number_input("Applicant Age", min_value=18, max_value=95, value=28)
            in_gender = st.selectbox("Gender", ["male", "female"])
            in_education = st.selectbox("Highest Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"], index=2)
            in_income = st.number_input("Annual Income ($)", min_value=5000, max_value=1000000, value=65000, step=2500)
            in_exp = st.slider("Employment Experience (Years)", min_value=0, max_value=40, value=4)
        with c_in2:
            in_home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"], index=2)
            in_amount = st.number_input("Requested Loan Amount ($)", min_value=500, max_value=50000, value=12000, step=500)
            in_intent = st.selectbox("Loan Purpose", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"], index=0)
            in_rate = st.slider("Loan Interest Rate (%)", min_value=4.0, max_value=25.0, value=11.5, step=0.1)
            in_credit = st.slider("Applicant Credit Score", min_value=390, max_value=850, value=710, step=5)
            
        c_in3, c_in4 = st.columns(2)
        with c_in3:
            in_cred_len = st.number_input("Credit History Length (Years)", min_value=0.0, max_value=30.0, value=6.0, step=0.5)
        with c_in4:
            in_default = st.selectbox("Previous Loan Default on File?", ["No", "Yes"], index=0)
            
        calc_lpi = in_amount / max(in_income, 1)
        st.caption(f"Computed Loan-to-Income Ratio: **{calc_lpi:.1%}**")
        
        run_sim = st.button("Evaluate Application Now", type="primary", use_container_width=True)
        
    with col_sim_out:
        st.markdown("##### 🎯 Underwriting Assessment")
        
        # Prepare dictionary
        applicant_payload = {
            "person_age": float(in_age),
            "person_gender": in_gender,
            "person_education": in_education,
            "person_income": float(in_income),
            "person_emp_exp": int(in_exp),
            "person_home_ownership": in_home,
            "loan_amnt": float(in_amount),
            "loan_intent": in_intent,
            "loan_int_rate": float(in_rate),
            "loan_percent_income": float(calc_lpi),
            "cb_person_cred_hist_length": float(in_cred_len),
            "credit_score": int(in_credit),
            "previous_loan_defaults_on_file": in_default
        }
        
        res = model_utils.predict_single_applicant(
            model, scaler, metrics["feature_columns"], applicant_payload
        )
        
        if res["is_approved"]:
            st.markdown(f"""
            <div class="predict-approved">
                <h2 style="color: #10b981; margin:0;">🎉 LOAN APPROVED</h2>
                <p style="font-size:14px; color:#cbd5e1; margin-top:6px;">Applicant meets decision tree credit criteria.</p>
                <h3 style="color:#ffffff; margin:10px 0 0 0;">Approval Confidence: {res['approval_probability']:.1%}</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="predict-rejected">
                <h2 style="color: #f43f5e; margin:0;">🛑 LOAN REJECTED</h2>
                <p style="font-size:14px; color:#cbd5e1; margin-top:6px;">Application failed credit thresholds.</p>
                <h3 style="color:#ffffff; margin:10px 0 0 0;">Rejection Confidence: {res['rejection_probability']:.1%}</h3>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("##### 🔍 Decision Reasoning & Risk Flags")
        for r in res["reasons"]:
            st.markdown(r)
            
        st.markdown("##### 💡 Advisory")
        if res["is_approved"]:
            st.success("Application qualifies under current risk rules. Standard documentation checks recommended.")
        else:
            if in_default == "Yes":
                st.error("Primary barrier is previous loan default record. No loans are approved with default records in this risk model.")
            elif calc_lpi > 0.30:
                st.warning("Consider lowering requested loan amount or adding collateral to reduce loan-to-income ratio below 25%.")
            else:
                st.info("Consider adjusting interest rate or reviewing secondary applicant credit variables.")


# ============================================================================
# TAB 5: REPORT & ARTIFACTS
# ============================================================================
with tab_report:
    st.markdown("### 📄 Academic Project Report & Methodological Documentation")
    
    col_rep1, col_rep2 = st.columns([1.5, 1])
    with col_rep1:
        st.markdown(r"""
        #### 1. Executive Summary
        When a customer applies for a loan, financial institutions must decide whether to approve or reject the application. In this project, a **Decision Tree model** is trained on **45,000 past loan applications** to automate this decision process based on applicant financial history.
        
        #### 2. Problem Statement Workflow
        1. **Load Data:** 45,000 applications, 14 feature variables.
        2. **Clean & Standardize:** Median imputation for 7 impossible ages (> 100); StandardScaler on Credit Score ($\mu=632.65, \sigma=50.29$).
        3. **Train Model:** Scikit-Learn `DecisionTreeClassifier(max_depth=5, random_state=42)`.
        4. **Evaluate Performance:** Stratified 80/20 test split (9,000 unseen applications), confusion matrix analysis.
        5. **Income Analysis:** Quartile grouping exploring approval dynamics across income brackets.
        
        #### 3. Empirical Results
        - **Accuracy:** 91.17% (vs. baseline no-information rate of 77.8%)
        - **Precision (Approved):** 85.7%
        - **Recall (Approved):** 72.3%
        - **Confusion Matrix:** 6,759 True Negatives, 1,446 True Positives, 241 False Approvals, 554 False Rejections.
        """)
        
    with col_rep2:
        st.markdown("#### 📥 Project Artifacts & Downloads")
        
        # Download DOCX report
        if os.path.exists("Predicting_Loan_Approval_Report.docx"):
            with open("Predicting_Loan_Approval_Report.docx", "rb") as f:
                docx_bytes = f.read()
            st.download_button(
                label="📄 Download Official DOCX Report (2 MB)",
                data=docx_bytes,
                file_name="Predicting_Loan_Approval_Report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
            
        # Download cleaned CSV
        if os.path.exists("loan_data.csv"):
            with open("loan_data.csv", "rb") as f:
                csv_bytes = f.read()
            st.download_button(
                label="📊 Download Cleaned Dataset (3.6 MB CSV)",
                data=csv_bytes,
                file_name="loan_data.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        # Download results JSON
        if os.path.exists("outputs/results.json"):
            with open("outputs/results.json", "rb") as f:
                res_bytes = f.read()
            st.download_button(
                label="⚙️ Download Metrics results.json",
                data=res_bytes,
                file_name="results.json",
                mime="application/json",
                use_container_width=True
            )
            
        st.markdown("""
        **Repository:**  
        `https://github.com/Saransh2005/clustering1.git`
        
        **Streamlit Cloud Ready:**  
        Main branch, Entrypoint: `app.py`
        """)

st.markdown("""
<div style="text-align:center; padding: 20px 0 10px 0; color:#64748b; font-size:12px;">
    Lovely Professional University • Project / Practical File • Saransh Singh (Reg: 12301472, Roll: 35, Sec: RK2P23FHB36)
</div>
""", unsafe_allow_html=True)
