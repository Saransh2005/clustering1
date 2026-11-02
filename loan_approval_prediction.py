# -*- coding: utf-8 -*-
"""
Predicting Loan Approval using a Decision Tree
==============================================
Name      : Saransh Singh
Reg. No.  : 12301472
Roll No.  : 35
Section   : RK2P23FHB36
Dataset   : Loan Approval Classification Dataset (Kaggle, 45,000 applications)
https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data

Tasks
1. Load the loan application dataset.
2. Handle missing values and standardize credit scores.
3. Train a Decision Tree model for loan approval prediction.
4. Evaluate the performance using a confusion matrix.
5. Visualize approval rates by income group.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")                       # save charts as files (no pop-up window)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

CSV_PATH = "loan_data.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
results = {}                                 # numbers saved for the report

def banner(text):
    print("\n" + "=" * 66 + "\n " + text + "\n" + "=" * 66)

# ---------------------------------------------------------------------------
# STEP 1 - Load the loan application dataset
# ---------------------------------------------------------------------------
banner("STEP 1: LOAD THE DATASET")
df = pd.read_csv(CSV_PATH)
print("Rows (applications) :", f"{len(df):,}")
print("Columns             :", df.shape[1])
print("Target column       : loan_status  (1 = approved, 0 = rejected)")
print("\nFirst 3 applications (columns shown downwards):")
print(df.head(3).T.to_string(header=False))

results["rows"] = len(df)
results["columns"] = df.shape[1]

# ---------------------------------------------------------------------------
# STEP 2 - Handle missing values and standardize credit scores
# ---------------------------------------------------------------------------
banner("STEP 2: HANDLE MISSING VALUES + STANDARDIZE CREDIT SCORE")
# 2a. Count blank (missing) cells
blank_cells = int(df.isnull().sum().sum())
print("2a. Blank cells in the file          :", blank_cells)

# 2b. Some ages are impossible (e.g. 144 years). Treat them as missing values.
bad_age = df["person_age"] > 100
print("2b. Applications with age above 100  :", int(bad_age.sum()))
df.loc[bad_age, "person_age"] = np.nan

# 2c. Fill every missing value: numbers -> median, text -> most common value
missing_total = int(df.isnull().sum().sum())
for col in df.columns:
    if df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
print("2c. Missing values filled with median:", missing_total,
      "| left after filling:", int(df.isnull().sum().sum()))

# 2d. Convert text columns into 0/1 columns (a model understands only numbers)
y = df["loan_status"]
X = pd.get_dummies(df.drop(columns="loan_status"), drop_first=True).astype(float)
print("2d. Columns after text -> number     :", X.shape[1])

# 2e. Split: 80 % to learn from (train), 20 % to test on (test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_test = X_train.copy(), X_test.copy()
print(f"2e. Training rows: {len(X_train):,}   Testing rows: {len(X_test):,}")

# 2f. Standardize credit score: (score - mean) / standard deviation
X_train_raw, X_test_raw = X_train.copy(), X_test.copy()   # un-scaled copies (for a check)
credit_before = X_train["credit_score"].copy()
mean_before, std_before = credit_before.mean(), credit_before.std()

scaler = StandardScaler()
# learn mean/std from the training rows, then use the same values on the test rows
X_train["credit_score"] = scaler.fit_transform(X_train[["credit_score"]])
X_test["credit_score"] = scaler.transform(X_test[["credit_score"]])
print(f"2f. Credit score BEFORE : mean = {mean_before:7.2f}, std = {std_before:6.2f}")
print(f"    Credit score AFTER  : mean = {X_train['credit_score'].mean():7.2f}, "
      f"std = {X_train['credit_score'].std():6.2f}")

results.update(blank_cells=blank_cells, bad_age=int(bad_age.sum()),
               train_rows=len(X_train), test_rows=len(X_test),
               features=X.shape[1], cs_mean=float(mean_before),
               cs_std=float(std_before))

# ---------------------------------------------------------------------------
# STEP 3 - Train the Decision Tree
# ---------------------------------------------------------------------------
banner("STEP 3: TRAIN THE DECISION TREE")
# max_depth=5 -> the tree asks at most 5 questions in a row.
# This keeps the tree simple and stops it from memorising the training data.
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
print("Model          : DecisionTreeClassifier(max_depth=5)")
print("Depth reached  :", model.get_depth())
print("Leaf nodes     :", model.get_n_leaves())

# Which columns did the tree use the most?
importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)
print("\nTop 5 most important columns:")
print(importance.head(5).round(3).to_string())
results["top_features"] = {k: float(v) for k, v in importance.head(8).items()}

# ---------------------------------------------------------------------------
# STEP 4 - Evaluate with a confusion matrix
# ---------------------------------------------------------------------------
banner("STEP 4: EVALUATE THE MODEL")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"Accuracy on test data : {accuracy:.2%}")
print("\nConfusion matrix:")
print(f"                      Predicted Rejected   Predicted Approved")
print(f"  Actual Rejected     {tn:>12,} (TN)      {fp:>12,} (FP)")
print(f"  Actual Approved     {fn:>12,} (FN)      {tp:>12,} (TP)")
print("\nClassification report:")
print(classification_report(y_test, y_pred,
                            target_names=["Rejected", "Approved"], digits=3))

precision = tp / (tp + fp)       # of all predicted "Approved", how many were right
recall = tp / (tp + fn)          # of all really approved, how many did we find
results.update(accuracy=float(accuracy), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
               precision=float(precision), recall=float(recall),
               f1=float(2 * precision * recall / (precision + recall)))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["Rejected", "Approved"], cmap="Blues", ax=ax)
ax.set_title(f"Confusion Matrix (accuracy {accuracy:.1%})", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# STEP 5 - Approval rate by income group
# ---------------------------------------------------------------------------
banner("STEP 5: APPROVAL RATE BY INCOME GROUP")
# Split applicants into 4 equal-sized income groups (quartiles)
names = ["Low", "Lower-Middle", "Upper-Middle", "High"]
df["income_group"], edges = pd.qcut(df["person_income"], q=4,
                                    labels=names, retbins=True)
table = df.groupby("income_group", observed=True).agg(
    applicants=("loan_status", "size"),
    approved=("loan_status", "sum"),
    approval_rate=("loan_status", "mean"))
table["approval_rate"] = table["approval_rate"] * 100
table["income_range"] = [f"{edges[i]:,.0f} - {edges[i + 1]:,.0f}" for i in range(4)]
print(table.round(1).to_string())

overall_rate = df["loan_status"].mean() * 100
print(f"\nOverall approval rate: {overall_rate:.1f} %")
income_rows = table.reset_index().astype({"income_group": str})
results["income_table"] = income_rows.to_dict("records")
results["overall_rate"] = float(overall_rate)

colors = ["#e03131", "#f59f00", "#74c0fc", "#1c7ed6"]
fig, ax = plt.subplots(figsize=(8, 5.5))
bars = ax.bar(names, table["approval_rate"], color=colors, edgecolor="black")
for bar, (_, row) in zip(bars, table.iterrows()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
            f"{row['approval_rate']:.1f}%", ha="center", fontweight="bold")
ax.axhline(overall_rate, color="black", ls="--", lw=1)
ax.text(3.45, overall_rate + 0.6, f"overall {overall_rate:.1f}%",
        ha="right", fontsize=9)
ax.set_xticks(range(4))
labels = [f"{n}\n{r}" for n, r in zip(names, table["income_range"])]
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Approval rate (%)")
ax.set_xlabel("Income group (annual income)")
ax.set_title("Loan Approval Rate by Income Group", fontweight="bold")
ax.set_ylim(0, table["approval_rate"].max() + 6)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/approval_by_income.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# EXTRA CHECK - what really separates approved from rejected in this data?
# ---------------------------------------------------------------------------
banner("EXTRA CHECK: WHAT DRIVES APPROVAL IN THIS DATASET?")
print("Approval rate (%) by previous default on file:")
by_default = df.groupby("previous_loan_defaults_on_file")["loan_status"].mean() * 100
print(by_default.round(1).to_string())
results["check"] = {}
for col in ["credit_score", "loan_percent_income"]:
    groups = pd.qcut(df[col], q=4, duplicates="drop")
    rates = (df.groupby(groups, observed=True)["loan_status"].mean() * 100).round(1)
    print(f"\nApproval rate (%) by {col} (4 equal groups, low -> high):")
    print(rates.to_string())
    results["check"][col] = [float(v) for v in rates.values]
results["check"]["default_yes"] = float(by_default["Yes"])
results["check"]["default_no"] = float(by_default["No"])

# Does standardizing change a Decision Tree? Train a second tree on the un-scaled data.
raw_model = DecisionTreeClassifier(max_depth=5, random_state=42)
raw_model.fit(X_train_raw, y_train)
same = bool((raw_model.predict(X_test_raw) == y_pred).all())
print("\nPredictions identical with and without scaling :", same)
results["scaling_same"] = same

print("\nAverage loan_percent_income of each income group (loan size / income):")
lpi = df.groupby("income_group", observed=True)["loan_percent_income"].mean()
print(lpi.round(2).to_string())
results["check"]["lpi_by_income"] = [float(v) for v in lpi.values]

# ---------------------------------------------------------------------------
# EXTRA CHARTS - tree picture, important columns, depth check, dashboard
# ---------------------------------------------------------------------------
banner("EXTRA: TREE PICTURE, FEATURE IMPORTANCE, DASHBOARD")
# (a) Picture of the top 3 levels of the tree
fig, ax = plt.subplots(figsize=(18, 8.5))
plot_tree(model, max_depth=3, feature_names=list(X.columns),
          class_names=["Rejected", "Approved"],
          filled=True, rounded=True, fontsize=11, ax=ax)
ax.set_title("Decision Tree - top 3 levels (blue = Approved, orange = Rejected)",
             fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/decision_tree.png", dpi=130)
plt.close(fig)

# (b) Most important columns
top = importance.head(8)[::-1]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(top.index, top.values, color="#1c7ed6")
ax.set_xlabel("Importance")
ax.set_title("Top 8 Most Important Columns", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150)
plt.close(fig)

# (c) Why depth 5?  Compare train / test accuracy for different depths
depths = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
train_acc, test_acc = [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, y_train)
    train_acc.append(m.score(X_train, y_train) * 100)
    test_acc.append(m.score(X_test, y_test) * 100)
results["depth_check"] = [{"depth": d, "train": round(a, 2), "test": round(b, 2)}
                          for d, a, b in zip(depths, train_acc, test_acc)]

# (d) Dashboard - everything on one page
fig = plt.figure(figsize=(17, 11.5), facecolor="#f8f9fa")
gs = fig.add_gridspec(3, 3, height_ratios=[0.42, 1.5, 1.5], hspace=0.5, wspace=0.25,
                      left=0.05, right=0.97, top=0.90, bottom=0.06)
fig.suptitle("LOAN APPROVAL PREDICTION - DASHBOARD", fontsize=21,
             fontweight="bold", y=0.975)

kpis = [("Applications", f"{len(df):,}", "#1c7ed6"),
        ("Overall approval rate", f"{overall_rate:.1f}%", "#1c7ed6"),
        ("Model accuracy (test)", f"{accuracy:.1%}", "#2f9e44"),
        ("Precision (approved)", f"{precision:.1%}", "#2f9e44"),
        ("Recall (approved)", f"{recall:.1%}", "#f59f00")]
inner = gs[0, :].subgridspec(1, 5, wspace=0.08)
for i, (label, value, color) in enumerate(kpis):
    ax = fig.add_subplot(inner[0, i])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_edgecolor(color); s.set_linewidth(2)
    ax.text(0.5, 0.62, value, ha="center", va="center", transform=ax.transAxes,
            fontsize=19, fontweight="bold", color=color)
    ax.text(0.5, 0.18, label, ha="center", va="center", transform=ax.transAxes,
            fontsize=10, color="#495057")

ax = fig.add_subplot(gs[1, 0])                       # approved vs rejected
counts = df["loan_status"].value_counts().sort_index()
ax.bar(["Rejected", "Approved"], counts.values, color=["#e03131", "#2f9e44"])
for i, v in enumerate(counts.values):
    ax.text(i, v + 400, f"{v:,}", ha="center", fontweight="bold")
ax.set_title("Approved vs rejected applications", fontweight="bold")
ax.set_ylabel("Applications")
ax.set_ylim(0, counts.max() * 1.12)

ax = fig.add_subplot(gs[1, 1])                       # credit score before / after
ax.hist(credit_before, bins=40, color="#74c0fc", edgecolor="white")
ax.set_xlabel("Credit score (original)")
ax.set_ylabel("Applications")
ax2 = ax.twiny()
ax2.hist(X_train["credit_score"], bins=40, histtype="step", color="#e03131", lw=1.8)
ax2.set_xlabel("Credit score (standardized, red outline)", color="#e03131")
ax.set_title("Credit score before and after standardizing", fontweight="bold", pad=34)

ax = fig.add_subplot(gs[1, 2])                       # approval by income
ax.bar(names, table["approval_rate"], color=colors, edgecolor="black")
for i, v in enumerate(table["approval_rate"]):
    ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")
ax.set_title("Approval rate by income group", fontweight="bold")
ax.set_ylabel("Approval rate (%)")
ax.set_ylim(0, table["approval_rate"].max() + 6)

ax = fig.add_subplot(gs[2, 0])                       # confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["Rejected", "Approved"],
    cmap="Blues", ax=ax, colorbar=False)
ax.set_title("Confusion matrix (test data)", fontweight="bold")

ax = fig.add_subplot(gs[2, 1])                       # important columns
ax.barh(top.index, top.values, color="#1c7ed6")
ax.tick_params(axis="y", labelsize=8)
ax.set_title("Most important columns", fontweight="bold")

ax = fig.add_subplot(gs[2, 2])                       # depth check
ax.plot(depths, train_acc, "o-", color="#e03131", label="Training accuracy")
ax.plot(depths, test_acc, "s-", color="#1c7ed6", label="Test accuracy")
ax.axvline(5, color="black", ls="--", lw=1)
ax.text(5.3, min(test_acc) + 1, "chosen depth = 5", fontsize=9)
ax.set_xticks(depths)
ax.set_xlabel("max_depth of the tree")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Why depth 5? (overfitting check)", fontweight="bold")
ax.legend(loc="lower right")

fig.savefig(f"{OUT_DIR}/dashboard.png", dpi=130, facecolor=fig.get_facecolor())
plt.close(fig)

with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("Saved: confusion_matrix.png, approval_by_income.png, decision_tree.png,")
print("       feature_importance.png, dashboard.png  ->  folder '" + OUT_DIR + "'")
print("\nDone.")
