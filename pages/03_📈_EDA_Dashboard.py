import streamlit as st
import pandas as pd
import plotly.express as px

# ==================================
# PAGE CONFIG
# ==================================

st.set_page_config(
    page_title="EDA Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==================================
# LOAD DATA
# ==================================

@st.cache_data
def load_data():
    return pd.read_csv(
        "data/processed/financial_fraud_feature_engineered_sample.csv"
    )

df = load_data()

# ==================================
# HEADER
# ==================================

st.markdown("""
<div class='hero'>
<h1>📊 Exploratory Data Analysis</h1>
<h3>Fraud Pattern Discovery & Behavioral Analytics</h3>
<p>
Analyze customer behavior, device activity,
payment methods and fraud trends.
</p>
</div>
""", unsafe_allow_html=True)

# ==================================
# AGE DISTRIBUTION
# ==================================

st.subheader("👥 Customer Age Distribution")

fig = px.histogram(
    df,
    x="customer_age",
    nbins=30,
    title="Customer Age Distribution"
)

fig.update_traces(
    marker_color="#3B82F6"
)

fig.update_layout(
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==================================
# FRAUD BY AGE
# ==================================

st.subheader("🚨 Fraud by Customer Age")

age_fraud = (
    df.groupby("customer_age")
    ["fraud_bool"]
    .sum()
    .reset_index()
)

fig = px.line(
    age_fraud,
    x="customer_age",
    y="fraud_bool",
    title="Fraud Cases by Age"
)

fig.update_traces(
    line_color="#EF4444"
)

fig.update_layout(
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==================================
# PAYMENT METHODS
# ==================================

st.subheader("💳 Payment Method Analysis")

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
    marker_color="#06B6D4"
)

fig.update_layout(
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==================================
# DEVICE ANALYSIS
# ==================================

st.subheader("📱 Device Operating System Analysis")

device_cols = [
    "device_os_1",
    "device_os_2",
    "device_os_3",
    "device_os_4"
]

device_data = pd.DataFrame({
    "Device OS": device_cols,
    "Count": [df[col].sum() for col in device_cols]
})

fig = px.bar(
    device_data,
    x="Device OS",
    y="Count",
    title="Device Distribution"
)

fig.update_traces(
    marker_color="#8B5CF6"
)

fig.update_layout(
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==================================
# MONTHLY FRAUD TREND
# ==================================

st.subheader("📅 Monthly Fraud Trend")

monthly_fraud = (
    df.groupby("month")
    ["fraud_bool"]
    .sum()
    .reset_index()
)

fig = px.line(
    monthly_fraud,
    x="month",
    y="fraud_bool",
    markers=True,
    title="Monthly Fraud Cases"
)

fig.update_traces(
    line_color="#F97316"
)

fig.update_layout(
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==================================
# CREDIT RISK DISTRIBUTION
# ==================================

st.subheader("🎯 Credit Risk Distribution")

fig = px.histogram(
    df,
    x="credit_risk_score",
    nbins=40,
    title="Credit Risk Score Distribution"
)

fig.update_traces(
    marker_color="#10B981"
)

fig.update_layout(
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==================================
# TRUST SCORE DISTRIBUTION
# ==================================

st.subheader("🛡️ Customer Trust Score")

fig = px.histogram(
    df,
    x="customer_trust_score",
    nbins=40,
    title="Customer Trust Score Distribution"
)

fig.update_traces(
    marker_color="#3B82F6"
)

fig.update_layout(
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==================================
# RISK VS TRUST
# ==================================

st.subheader("⚖️ Risk Score vs Trust Score")

sample_df = df.sample(
    min(5000, len(df)),
    random_state=42
)

fig = px.scatter(
    sample_df,
    x="customer_trust_score",
    y="credit_risk_score",
    color="fraud_bool",
    title="Risk vs Trust Relationship"
)

fig.update_layout(
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==================================
# TRANSACTION VELOCITY
# ==================================

st.subheader("⚡ Transaction Velocity Analysis")

fig = px.box(
    sample_df,
    x="fraud_bool",
    y="transaction_velocity_score",
    title="Velocity Score vs Fraud"
)

fig.update_layout(
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==================================
# TOP RISKY RECORDS
# ==================================

st.subheader("🚨 Top High-Risk Customers")

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
            "fraud_bool"
        ]
    ],
    width="stretch"
)

# ==================================
# SUMMARY
# ==================================

st.markdown("---")

st.success("""
EDA Insights Generated Successfully

✔ Customer Age Analysis

✔ Fraud Trend Analysis

✔ Device Analytics

✔ Payment Analytics

✔ Risk Score Analysis

✔ Velocity Analysis

✔ Trust Score Analysis
""")