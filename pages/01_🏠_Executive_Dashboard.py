import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(
        "data/processed/financial_fraud_feature_engineered_sample.csv"
    )

df = load_data()

# -----------------------------
# HERO SECTION
# -----------------------------
st.markdown("""
<div class="hero">
<h1>🛡️ FraudShield AI</h1>
<h3>Enterprise Fraud Detection & Risk Intelligence Platform</h3>
<p>
Real-Time Monitoring • Machine Learning • Risk Scoring • Fraud Investigation • Graph Analytics
</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# KPI METRICS
# -----------------------------
total_transactions = len(df)

fraud_cases = int(df["fraud_bool"].sum())

fraud_rate = round(
    (fraud_cases / total_transactions) * 100,
    2
)

avg_risk = round(
    df["credit_risk_score"].mean(),
    2
)

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.metric(
        "📄 Transactions",
        f"{total_transactions:,}"
    )

with c2:
    st.metric(
        "🚨 Fraud Cases",
        f"{fraud_cases:,}"
    )

with c3:
    st.metric(
        "⚠️ Fraud Rate",
        f"{fraud_rate}%"
    )

with c4:
    st.metric(
        "🎯 Avg Risk Score",
        avg_risk
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# CHARTS ROW 1
# -----------------------------
col1,col2 = st.columns(2)

with col1:

    fraud_counts = (
        df["fraud_bool"]
        .value_counts()
        .reset_index()
    )

    fraud_counts.columns = [
        "Fraud",
        "Count"
    ]

    fig = px.pie(
        fraud_counts,
        names="Fraud",
        values="Count",
        title="Fraud Distribution",
        hole=0.55
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

with col2:

    fig = px.histogram(
        df,
        x="credit_risk_score",
        nbins=30,
        title="Credit Risk Distribution"
    )

    fig.update_traces(
        marker_color="#3B82F6"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

# -----------------------------
# CHARTS ROW 2
# -----------------------------
col1,col2 = st.columns(2)

with col1:

    payment_cols = [
        "payment_type_1",
        "payment_type_2",
        "payment_type_3",
        "payment_type_4"
    ]

    payment_data = pd.DataFrame({
        "Payment Type": payment_cols,
        "Count": [df[col].sum() for col in payment_cols]
    })

    fig = px.bar(
        payment_data,
        x="Payment Type",
        y="Count",
        title="Payment Method Usage"
    )

    fig.update_traces(
        marker_color="#3B82F6"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

with col2:

    device_cols = [
        "device_os_1",
        "device_os_2",
        "device_os_3",
        "device_os_4"
    ]

    device_data = pd.DataFrame({
        "Device": device_cols,
        "Count": [df[col].sum() for col in device_cols]
    })

    fig = px.bar(
        device_data,
        x="Device",
        y="Count",
        title="Device OS Distribution"
    )

    fig.update_traces(
        marker_color="#06B6D4"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

# -----------------------------
# CHARTS ROW 3
# -----------------------------
col1,col2 = st.columns(2)

with col1:

    fraud_month = (
        df.groupby("month")
        ["fraud_bool"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        fraud_month,
        x="month",
        y="fraud_bool",
        title="Monthly Fraud Trend"
    )

    fig.update_traces(
        line_color="#EF4444"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

with col2:

    fig = px.scatter(
        df.sample(5000),
        x="customer_trust_score",
        y="credit_risk_score",
        color="fraud_bool",
        title="Trust Score vs Risk Score"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

# -----------------------------
# TOP HIGH RISK RECORDS
# -----------------------------
st.markdown("## 🚨 Top High Risk Transactions")

top_risk = (
    df.sort_values(
        "credit_risk_score",
        ascending=False
    )
    .head(20)
)

st.dataframe(
    top_risk[
        [
            "credit_risk_score",
            "customer_trust_score",
            "transaction_velocity_score",
            "foreign_request",
            "fraud_bool"
        ]
    ],
    width="stretch"
)

# -----------------------------
# PROJECT SUMMARY
# -----------------------------
st.markdown("""
### 📌 Executive Summary

- Total Transactions Processed
- Fraud Detection Analytics
- Risk Intelligence Monitoring
- Device & Payment Insights
- Monthly Fraud Trends
- Customer Trust Analysis
- High Risk Transaction Monitoring

FraudShield AI provides enterprise-grade fraud detection using Machine Learning, Statistical Analysis, Graph Analytics and Real-Time Monitoring.
""")