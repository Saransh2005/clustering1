# -*- coding: utf-8 -*-
"""
Visualization Utilities for Loan Approval Dashboard
===================================================
Author      : Saransh Singh (Reg: 12301472, Roll: 35, Section: RK2P23FHB36)
Project     : Predicting Loan Approval using Decision Tree Classification
Institution : Lovely Professional University (LPU)
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Theme palette matching modern fintech dark mode
DARK_BG = "#0f172a"
CARD_BG = "#1e293b"
TEXT_COLOR = "#f8fafc"
ACCENT_BLUE = "#38bdf8"
ACCENT_GREEN = "#10b981"
ACCENT_RED = "#f43f5e"
ACCENT_AMBER = "#f59e0b"
ACCENT_PURPLE = "#a855f7"

def apply_base_layout(fig, title_text="", height=340):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", family="Plus Jakarta Sans, sans-serif"),
        margin=dict(l=30, r=30, t=45, b=30),
        height=height,
        title=dict(text=f"<b>{title_text}</b>", font=dict(size=14, color="#f8fafc")) if title_text else None
    )
    fig.update_xaxes(gridcolor="#334155", zerolinecolor="#334155")
    fig.update_yaxes(gridcolor="#334155", zerolinecolor="#334155")
    return fig


def fig_approval_donut(df):
    """Donut chart showing proportion of approved vs rejected loans."""
    counts = df["loan_status"].value_counts().sort_index()
    labels = ["Rejected (0)", "Approved (1)"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    colors = [ACCENT_RED, ACCENT_GREEN]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.58,
        marker=dict(colors=colors, line=dict(color="#0f172a", width=2)),
        textinfo="label+percent",
        hoverinfo="label+value+percent",
        textfont=dict(size=13, color="#ffffff", family="Plus Jakarta Sans")
    )])
    apply_base_layout(fig, title_text="Loan Application Status Distribution", height=320)
    fig.update_layout(
        showlegend=False,
        annotations=[dict(text=f"<b>{len(df):,}</b><br><span style='font-size:11px;color:#94a3b8;'>Total Apps</span>",
                          x=0.5, y=0.5, font_size=16, font_color="#ffffff", showarrow=False)]
    )
    return fig


def fig_income_approval_bars(income_table, overall_rate=22.2):
    """Bar chart showing approval rate across income quartile groups."""
    df_plot = income_table.reset_index()
    colors = ["#f43f5e", "#f59e0b", "#38bdf8", "#6366f1"]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_plot["income_group"].astype(str),
        y=df_plot["approval_rate_pct"],
        text=[f"<b>{val:.1f}%</b><br><span style='font-size:10px;'>{rng}</span>" 
              for val, rng in zip(df_plot["approval_rate_pct"], df_plot["income_range"])],
        textposition="outside",
        marker=dict(color=colors, line=dict(color="#0f172a", width=1.5)),
        hovertext=[f"Group: {g}<br>Range: {r}<br>Applicants: {a:,}<br>Approved: {app:,} ({rate:.1f}%)"
                   for g, r, a, app, rate in zip(df_plot["income_group"], df_plot["income_range"],
                                                 df_plot["applicants"], df_plot["approved"], df_plot["approval_rate_pct"])],
        hoverinfo="text"
    ))
    
    # Overall benchmark line
    fig.add_hline(
        y=overall_rate,
        line_dash="dash",
        line_color="#94a3b8",
        annotation_text=f"Overall Avg ({overall_rate:.1f}%)",
        annotation_position="top right",
        annotation_font=dict(size=11, color="#cbd5e1")
    )
    
    apply_base_layout(fig, title_text="Approval Rate by Income Quartile Group", height=340)
    fig.update_yaxes(title_text="Approval Rate (%)", range=[0, max(df_plot["approval_rate_pct"]) + 10])
    fig.update_xaxes(title_text="Income Group (Annual Income Range)")
    return fig


def fig_confusion_matrix(metrics):
    """Interactive heatmap confusion matrix with business labels."""
    tn, fp, fn, tp = metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]
    matrix = [[tn, fp], [fn, tp]]
    
    annotations = [
        [f"<b>TN: {tn:,}</b><br><span style='font-size:11px;'>Correctly Rejected<br>(96.6% Specificity)</span>",
         f"<b>FP: {fp:,}</b><br><span style='font-size:11px;'>False Approval<br>(Risk of Default)</span>"],
        [f"<b>FN: {fn:,}</b><br><span style='font-size:11px;'>False Rejection<br>(Missed Good Client)</span>",
         f"<b>TP: {tp:,}</b><br><span style='font-size:11px;'>Correctly Approved<br>(72.3% Recall)</span>"]
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=["Predicted Rejected", "Predicted Approved"],
        y=["Actual Rejected", "Actual Approved"],
        text=annotations,
        texttemplate="%{text}",
        textfont=dict(size=13, color="#ffffff"),
        colorscale="Blues",
        showscale=False
    ))
    
    apply_base_layout(fig, title_text=f"Confusion Matrix (Test Set N=9,000 | Acc {metrics['accuracy']:.2%})", height=340)
    fig.update_xaxes(title_text="Model Prediction", side="bottom")
    fig.update_yaxes(title_text="Actual Ground Truth", autorange="reversed")
    return fig


def fig_feature_importance(feat_series, top_n=8):
    """Horizontal bar chart for top feature importances."""
    top = feat_series.head(top_n).iloc[::-1]
    
    clean_labels = []
    for col in top.index:
        c = col.replace("previous_loan_defaults_on_file_Yes", "Prior Default (Yes)")
        c = c.replace("loan_percent_income", "Loan-to-Income %")
        c = c.replace("loan_int_rate", "Loan Interest Rate")
        c = c.replace("person_income", "Annual Income")
        c = c.replace("person_home_ownership_", "Home: ")
        c = c.replace("loan_intent_", "Intent: ")
        c = c.replace("person_education_", "Education: ")
        c = c.replace("cb_person_cred_hist_length", "Credit Hist Length")
        c = c.replace("credit_score", "Credit Score")
        clean_labels.append(c)
        
    fig = go.Figure(go.Bar(
        x=top.values,
        y=clean_labels,
        orientation="h",
        text=[f"<b>{v:.1%}</b>" for v in top.values],
        textposition="outside",
        marker=dict(
            color=top.values,
            colorscale=[[0, "#38bdf8"], [1, "#6366f1"]],
            line=dict(color="#0f172a", width=1)
        )
    ))
    apply_base_layout(fig, title_text=f"Top {top_n} Decision Tree Feature Drivers", height=340)
    fig.update_xaxes(title_text="Gini Feature Importance Share", range=[0, max(top.values) * 1.25])
    return fig


def fig_depth_overfitting(depth_df):
    """Train vs Test accuracy line chart showing depth 5 selection."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=depth_df["depth"],
        y=depth_df["train_acc"],
        mode="lines+markers",
        name="Training Accuracy",
        line=dict(color=ACCENT_RED, width=2.5),
        marker=dict(size=6)
    ))
    fig.add_trace(go.Scatter(
        x=depth_df["depth"],
        y=depth_df["test_acc"],
        mode="lines+markers",
        name="Test Accuracy",
        line=dict(color=ACCENT_BLUE, width=2.5),
        marker=dict(size=6)
    ))
    
    # Optimal depth 5 vertical line
    fig.add_vline(
        x=5,
        line_dash="dash",
        line_color="#10b981",
        line_width=2,
        annotation_text="Selected Depth = 5 (Optimal)",
        annotation_position="bottom right",
        annotation_font=dict(color="#10b981", size=11)
    )
    
    apply_base_layout(fig, title_text="Model Complexity & Overfitting Check (max_depth 1-20)", height=340)
    fig.update_xaxes(title_text="Tree Maximum Depth (max_depth)", tickmode="array", tickvals=list(depth_df["depth"]))
    fig.update_yaxes(title_text="Accuracy (%)")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#cbd5e1")))
    return fig


def fig_credit_score_standardization(raw_scores):
    """Histogram of raw credit score distribution."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=raw_scores,
        name="Original Score (μ=632.6, σ=50.3)",
        marker=dict(color="#38bdf8", opacity=0.7),
        nbinsx=40
    ))
    apply_base_layout(fig, title_text="Credit Score Distribution (Original Raw Values)", height=320)
    fig.update_xaxes(title_text="Credit Score Value")
    fig.update_yaxes(title_text="Application Count")
    return fig


def fig_categorical_approval(df, cat_col, title):
    """Bar chart showing approval rate by any categorical column."""
    grp = df.groupby(cat_col, observed=True).agg(
        total=("loan_status", "size"),
        approved=("loan_status", "sum"),
        rate=("loan_status", "mean")
    ).reset_index()
    grp["rate_pct"] = grp["rate"] * 100
    grp = grp.sort_values(by="rate_pct", ascending=False)
    
    fig = go.Figure(go.Bar(
        x=grp[cat_col],
        y=grp["rate_pct"],
        text=[f"<b>{r:.1f}%</b><br>({a:,}/{t:,})" for r, a, t in zip(grp["rate_pct"], grp["approved"], grp["total"])],
        textposition="outside",
        marker=dict(
            color=grp["rate_pct"],
            colorscale=[[0, "#38bdf8"], [1, "#10b981"]],
            line=dict(color="#0f172a", width=1)
        )
    ))
    apply_base_layout(fig, title_text=title, height=320)
    fig.update_yaxes(title_text="Approval Rate (%)", range=[0, max(grp["rate_pct"]) * 1.25])
    return fig


def fig_scatter_income_loan(df):
    """Scatter plot of Annual Income vs Loan Amount sampled for performance."""
    sample_df = df.sample(n=min(3000, len(df)), random_state=42).copy()
    sample_df["Status"] = sample_df["loan_status"].map({1: "Approved", 0: "Rejected"})
    
    fig = px.scatter(
        sample_df,
        x="person_income",
        y="loan_amnt",
        color="Status",
        color_discrete_map={"Approved": ACCENT_GREEN, "Rejected": ACCENT_RED},
        opacity=0.65,
        hover_data=["person_age", "loan_int_rate", "loan_percent_income", "credit_score"]
    )
    apply_base_layout(fig, title_text="Income vs. Requested Loan Amount (Sampled N=3,000)", height=340)
    fig.update_xaxes(title_text="Annual Income ($)", range=[0, 200000])
    fig.update_yaxes(title_text="Requested Loan Amount ($)", range=[0, 35000])
    return fig
