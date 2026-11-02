# -*- coding: utf-8 -*-
"""
Visualization Utilities for Loan Approval Dashboard
===================================================
Author      : Devangi Yadav (Reg: 12300822, Roll: 35, Section: RK2P23FHB35)
Project     : Predicting Loan Approval using Decision Tree Classification
Institution : Lovely Professional University (LPU)
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Clean Minimalist Dark Palette
DARK_BG = "#0f172a"
CARD_BG = "#1e293b"
ACCENT_BLUE = "#38bdf8"
ACCENT_GREEN = "#10b981"
ACCENT_RED = "#f43f5e"
ACCENT_AMBER = "#f59e0b"

def apply_base_layout(fig, title_text="", height=320):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", family="Plus Jakarta Sans, sans-serif", size=12),
        margin=dict(l=20, r=20, t=40, b=25),
        height=height,
        title=dict(text=f"<b>{title_text}</b>", font=dict(size=14, color="#f8fafc")) if title_text else None
    )
    fig.update_xaxes(gridcolor="#334155", zerolinecolor="#334155")
    fig.update_yaxes(gridcolor="#334155", zerolinecolor="#334155")
    return fig


def fig_approval_donut(df):
    """Clean Donut chart showing loan approval distribution."""
    counts = df["loan_status"].value_counts().sort_index()
    labels = ["Rejected", "Approved"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    colors = [ACCENT_RED, ACCENT_GREEN]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color="#0f172a", width=2)),
        textinfo="label+percent",
        hoverinfo="label+value+percent",
        textfont=dict(size=12, color="#ffffff")
    )])
    apply_base_layout(fig, title_text="Overall Application Status", height=280)
    fig.update_layout(
        showlegend=False,
        annotations=[dict(text=f"<b>{len(df):,}</b><br><span style='font-size:11px;color:#94a3b8;'>Applications</span>",
                          x=0.5, y=0.5, font_size=15, font_color="#ffffff", showarrow=False)]
    )
    return fig


def fig_income_approval_bars(income_table, overall_rate=22.2):
    """Clean bar chart for approval rates across income quartile groups."""
    df_plot = income_table.reset_index()
    colors = ["#f43f5e", "#f59e0b", "#38bdf8", "#6366f1"]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_plot["income_group"].astype(str),
        y=df_plot["approval_rate_pct"],
        text=[f"<b>{val:.1f}%</b>" for val in df_plot["approval_rate_pct"]],
        textposition="outside",
        marker=dict(color=colors, line=dict(color="#0f172a", width=1.5)),
        hovertext=[f"<b>{g} Income</b><br>Range: {r}<br>Applicants: {a:,}<br>Approved: {app:,} ({rate:.1f}%)"
                   for g, r, a, app, rate in zip(df_plot["income_group"], df_plot["income_range"],
                                                 df_plot["applicants"], df_plot["approved"], df_plot["approval_rate_pct"])],
        hoverinfo="text"
    ))
    
    fig.add_hline(
        y=overall_rate,
        line_dash="dash",
        line_color="#94a3b8",
        annotation_text=f"Avg: {overall_rate:.1f}%",
        annotation_position="top right",
        annotation_font=dict(size=11, color="#cbd5e1")
    )
    
    apply_base_layout(fig, title_text="Approval Rate by Income Group", height=280)
    fig.update_yaxes(title_text="Approval Rate (%)", range=[0, max(df_plot["approval_rate_pct"]) + 8])
    fig.update_xaxes(title_text="")
    return fig


def fig_confusion_matrix(metrics):
    """Clean interactive confusion matrix."""
    tn, fp, fn, tp = metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]
    matrix = [[tn, fp], [fn, tp]]
    
    annotations = [
        [f"<b>TN: {tn:,}</b><br><span style='font-size:11px;'>Correct Rejection</span>",
         f"<b>FP: {fp:,}</b><br><span style='font-size:11px;'>Wrong Approval</span>"],
        [f"<b>FN: {fn:,}</b><br><span style='font-size:11px;'>Missed Approval</span>",
         f"<b>TP: {tp:,}</b><br><span style='font-size:11px;'>Correct Approval</span>"]
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=["Pred. Rejected", "Pred. Approved"],
        y=["Act. Rejected", "Act. Approved"],
        text=annotations,
        texttemplate="%{text}",
        textfont=dict(size=12, color="#ffffff"),
        colorscale="Blues",
        showscale=False
    ))
    
    apply_base_layout(fig, title_text=f"Confusion Matrix (Accuracy {metrics['accuracy']:.1%})", height=280)
    fig.update_xaxes(title_text="", side="bottom")
    fig.update_yaxes(title_text="", autorange="reversed")
    return fig


def fig_feature_importance(feat_series, top_n=5):
    """Clean horizontal bar chart for top feature drivers."""
    top = feat_series.head(top_n).iloc[::-1]
    
    clean_labels = []
    for col in top.index:
        c = col.replace("previous_loan_defaults_on_file_Yes", "Prior Default (Yes)")
        c = c.replace("loan_percent_income", "Loan-to-Income %")
        c = c.replace("loan_int_rate", "Loan Interest Rate")
        c = c.replace("person_income", "Annual Income")
        c = c.replace("person_home_ownership_RENT", "Home: RENT")
        c = c.replace("person_home_ownership_OWN", "Home: OWN")
        clean_labels.append(c)
        
    fig = go.Figure(go.Bar(
        x=top.values * 100,
        y=clean_labels,
        orientation="h",
        text=[f"<b>{v*100:.1f}%</b>" for v in top.values],
        textposition="outside",
        marker=dict(color="#38bdf8", line=dict(color="#0f172a", width=1))
    ))
    apply_base_layout(fig, title_text=f"Top {top_n} Factors Influencing Approval", height=280)
    fig.update_xaxes(title_text="Importance Share (%)", range=[0, max(top.values * 100) * 1.25])
    fig.update_yaxes(title_text="")
    return fig


def fig_depth_overfitting(depth_df):
    """Clean line chart of model depth vs accuracy."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=depth_df["depth"],
        y=depth_df["train_acc"],
        mode="lines+markers",
        name="Training Accuracy",
        line=dict(color=ACCENT_RED, width=2),
        marker=dict(size=5)
    ))
    fig.add_trace(go.Scatter(
        x=depth_df["depth"],
        y=depth_df["test_acc"],
        mode="lines+markers",
        name="Test Accuracy",
        line=dict(color=ACCENT_BLUE, width=2),
        marker=dict(size=5)
    ))
    
    fig.add_vline(
        x=5,
        line_dash="dash",
        line_color="#10b981",
        line_width=1.5,
        annotation_text="Chosen Depth = 5",
        annotation_position="bottom right",
        annotation_font=dict(color="#10b981", size=10)
    )
    
    apply_base_layout(fig, title_text="Tree Depth vs. Accuracy (Overfitting Check)", height=280)
    fig.update_xaxes(title_text="Tree Depth", tickmode="array", tickvals=list(depth_df["depth"]))
    fig.update_yaxes(title_text="Accuracy (%)")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig
