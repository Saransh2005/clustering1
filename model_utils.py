# -*- coding: utf-8 -*-
"""
Model Utilities and Analytics for Loan Approval Prediction
==========================================================
Author      : Devangi Yadav (Reg: 12300822, Roll: 35, Section: RK2P23FHB35)
Project     : Predicting Loan Approval using Decision Tree Classification
Institution : Lovely Professional University (LPU)
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

CSV_PATH_DEFAULT = "loan_data.csv"

def load_and_preprocess_data(csv_path=CSV_PATH_DEFAULT):
    """
    Load loan_data.csv, clean anomalies (age > 100), impute missing values,
    and compute income quartiles.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
        
    df_raw = pd.read_csv(csv_path)
    df = df_raw.copy()
    
    # Treat impossible ages (> 100) as missing values
    bad_age_mask = df["person_age"] > 100
    bad_age_count = int(bad_age_mask.sum())
    df.loc[bad_age_mask, "person_age"] = np.nan
    
    # Impute missing values (median for numeric, mode for categorical)
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
                
    # Income quartiles
    group_names = ["Low", "Lower-Middle", "Upper-Middle", "High"]
    df["income_group"], edges = pd.qcut(df["person_income"], q=4, labels=group_names, retbins=True)
    income_ranges = [f"${edges[i]:,.0f} - ${edges[i+1]:,.0f}" for i in range(4)]
    
    income_table = df.groupby("income_group", observed=True).agg(
        applicants=("loan_status", "size"),
        approved=("loan_status", "sum"),
        approval_rate=("loan_status", "mean")
    )
    income_table["approval_rate_pct"] = income_table["approval_rate"] * 100
    income_table["income_range"] = income_ranges
    
    return df, bad_age_count, income_table


def train_model(df):
    """
    Prepare features, perform stratified 80/20 train/test split,
    standardize credit score, and train DecisionTreeClassifier(max_depth=5).
    """
    feature_df = df.drop(columns=["loan_status", "income_group"], errors="ignore")
    y = df["loan_status"].astype(int)
    
    # One-hot encoding for text columns (drop_first=True)
    X = pd.get_dummies(feature_df, drop_first=True).astype(float)
    feature_columns = list(X.columns)
    
    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_test = X_train.copy(), X_test.copy()
    
    # Credit score standardization
    scaler = StandardScaler()
    cs_raw_train = X_train["credit_score"].copy()
    cs_mean_raw = float(cs_raw_train.mean())
    cs_std_raw = float(cs_raw_train.std())
    
    X_train["credit_score"] = scaler.fit_transform(X_train[["credit_score"]])
    X_test["credit_score"] = scaler.transform(X_test[["credit_score"]])
    
    # Decision tree training
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions & metrics
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Feature importances
    feat_imp = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
    
    # Depth complexity curve
    depths = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
    depth_curve = []
    for d in depths:
        m = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, y_train)
        train_s = m.score(X_train, y_train) * 100
        test_s = m.score(X_test, y_test) * 100
        depth_curve.append({"depth": d, "train_acc": train_s, "test_acc": test_s})
    depth_df = pd.DataFrame(depth_curve)
    
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_test": len(y_test),
        "cs_mean_raw": cs_mean_raw,
        "cs_std_raw": cs_std_raw,
        "feature_importances": feat_imp,
        "depth_df": depth_df,
        "feature_columns": feature_columns
    }
    
    return model, scaler, metrics, (X_train, X_test, y_train, y_test, y_pred, y_proba)


def predict_single_applicant(model, scaler, feature_columns, applicant_dict):
    """
    Predict single applicant loan approval with transparent decision reasoning.
    """
    input_df = pd.DataFrame([applicant_dict])
    
    if "loan_percent_income" not in input_df.columns or pd.isnull(input_df["loan_percent_income"].iloc[0]):
        input_df["loan_percent_income"] = input_df["loan_amnt"] / max(input_df["person_income"].iloc[0], 1.0)
        
    input_df["credit_score"] = scaler.transform(input_df[["credit_score"]])
    encoded_df = pd.get_dummies(input_df, drop_first=True)
    
    full_row = pd.DataFrame(0.0, index=[0], columns=feature_columns)
    for col in encoded_df.columns:
        if col in full_row.columns:
            full_row[col] = float(encoded_df[col].iloc[0])
            
    pred = int(model.predict(full_row)[0])
    prob_approved = float(model.predict_proba(full_row)[0][1])
    
    reasons = []
    if applicant_dict.get("previous_loan_defaults_on_file") == "Yes":
        reasons.append("⚠️ **Prior Default on Record**: Instant rejection trigger at tree root.")
    else:
        reasons.append("✅ **No Prior Defaults**: Passed initial credit history check.")
        
    lpi = applicant_dict.get("loan_percent_income", applicant_dict["loan_amnt"] / max(applicant_dict["person_income"], 1))
    if lpi > 0.35:
        reasons.append(f"⚠️ **High Loan-to-Income ({lpi:.1%})**: Loan requested is high relative to income.")
    else:
        reasons.append(f"✅ **Healthy Loan-to-Income ({lpi:.1%})**: Loan size is within safe bounds.")
        
    rate = applicant_dict.get("loan_int_rate", 10.0)
    if rate > 14.0:
        reasons.append(f"⚠️ **Higher Interest Rate ({rate:.1f}%)**: Indicates higher evaluated risk tier.")
    else:
        reasons.append(f"✅ **Moderate Interest Rate ({rate:.1f}%)**: Standard low-risk rate.")
        
    return {
        "prediction": pred,
        "is_approved": (pred == 1),
        "approval_probability": prob_approved,
        "rejection_probability": 1.0 - prob_approved,
        "reasons": reasons
    }
