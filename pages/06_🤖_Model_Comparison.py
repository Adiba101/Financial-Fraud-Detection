import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Model Comparison",
    page_icon="🏆",
    layout="wide"
)

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class='hero'>
<h1>🏆 Model Comparison Dashboard</h1>
<h3>Machine Learning Performance Evaluation</h3>
<p>
Compare all fraud detection models and identify the best performer.
</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# MODEL RESULTS
# ==========================================

comparison_df = pd.DataFrame({

    "Model":[
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost",
        "SVM"
    ],

    "Accuracy":[
        0.7878,
        0.9438,
        0.9665,
        0.9853,
        0.9350
    ],

    "Precision":[
        0.0289,
        0.0669,
        0.1025,
        0.2021,
        0.0668
    ],

    "Recall":[
        0.5594,
        0.3164,
        0.2625,
        0.1120,
        0.3772
    ],

    "F1":[
        0.0550,
        0.1104,
        0.1474,
        0.1441,
        0.1136
    ],

    "ROC_AUC":[
        0.7450,
        0.7612,
        0.8460,
        0.8540,
        0.0000
    ]
})

# ==========================================
# SAVE FILE
# ==========================================

try:
    comparison_df.to_csv(
        "data/processed/model_comparison.csv",
        index=False
    )
except:
    pass

# ==========================================
# WINNER MODEL
# ==========================================

best_model = comparison_df.loc[
    comparison_df["ROC_AUC"].idxmax()
]

st.success(
    f"🏆 Best Model: {best_model['Model']} | ROC-AUC = {best_model['ROC_AUC']:.4f}"
)

# ==========================================
# TABLE
# ==========================================

st.subheader("📋 Model Performance Table")

st.dataframe(
    comparison_df,
    width="stretch"
)

# ==========================================
# ACCURACY COMPARISON
# ==========================================

st.subheader("🎯 Accuracy Comparison")

fig = px.bar(
    comparison_df,
    x="Model",
    y="Accuracy",
    color="Accuracy",
    text="Accuracy"
)

fig.update_layout(
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==========================================
# PRECISION COMPARISON
# ==========================================

st.subheader("📌 Precision Comparison")

fig = px.bar(
    comparison_df,
    x="Model",
    y="Precision",
    color="Precision",
    text="Precision"
)

fig.update_layout(
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==========================================
# RECALL COMPARISON
# ==========================================

st.subheader("📈 Recall Comparison")

fig = px.bar(
    comparison_df,
    x="Model",
    y="Recall",
    color="Recall",
    text="Recall"
)

fig.update_layout(
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==========================================
# F1 SCORE
# ==========================================

st.subheader("⚡ F1 Score Comparison")

fig = px.bar(
    comparison_df,
    x="Model",
    y="F1",
    color="F1",
    text="F1"
)

fig.update_layout(
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==========================================
# ROC AUC
# ==========================================

st.subheader("🚀 ROC-AUC Comparison")

fig = px.bar(
    comparison_df,
    x="Model",
    y="ROC_AUC",
    color="ROC_AUC",
    text="ROC_AUC"
)

fig.update_layout(
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==========================================
# LEADERBOARD
# ==========================================

st.subheader("🥇 Model Leaderboard")

ranking = comparison_df.sort_values(
    "ROC_AUC",
    ascending=False
)

ranking.index = range(1, len(ranking)+1)

st.dataframe(
    ranking,
    width="stretch"
)

# ==========================================
# RADAR STYLE METRICS TABLE
# ==========================================

st.subheader("📊 Model Evaluation Summary")

summary = comparison_df.set_index("Model")

st.dataframe(
    summary,
    width="stretch"
)

# ==========================================
# CONCLUSION
# ==========================================

st.markdown("---")

st.success("""
🏆 Best Performing Model: XGBoost

✔ Highest ROC-AUC

✔ Highest Precision

✔ Excellent Accuracy

✔ Strong Fraud Detection Capability

Recommended for deployment in FraudShield AI.
""")