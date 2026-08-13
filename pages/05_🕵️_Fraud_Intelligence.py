import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Fraud Intelligence",
    page_icon="🕵️",
    layout="wide"
)

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv(
        "data/processed/financial_fraud_feature_engineered_sample.csv"
    )

df = load_data()

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class='hero'>
<h1>🕵️ Fraud Intelligence Center</h1>
<h3>Fraud Investigation & Risk Intelligence</h3>
<p>
Identify suspicious patterns, risky customers,
foreign requests and fraud indicators.
</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# KPI ROW
# ==========================================

fraud_cases = int(df["fraud_bool"].sum())

foreign_requests = int(df["foreign_request"].sum())

high_risk = len(
    df[df["credit_risk_score"] > 150]
)

avg_trust = round(
    df["customer_trust_score"].mean(),
    2
)

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.metric(
        "🚨 Fraud Cases",
        f"{fraud_cases:,}"
    )

with c2:
    st.metric(
        "🌍 Foreign Requests",
        f"{foreign_requests:,}"
    )

with c3:
    st.metric(
        "⚠️ High Risk Users",
        f"{high_risk:,}"
    )

with c4:
    st.metric(
        "🛡 Avg Trust Score",
        avg_trust
    )

# ==========================================
# FOREIGN REQUEST ANALYSIS
# ==========================================

st.subheader("🌍 Foreign Request Analysis")

foreign_df = (
    df["foreign_request"]
    .value_counts()
    .reset_index()
)

foreign_df.columns = [
    "Foreign Request",
    "Count"
]

fig = px.pie(
    foreign_df,
    names="Foreign Request",
    values="Count",
    hole=0.55
)

fig.update_layout(
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==========================================
# HIGH RISK CUSTOMERS
# ==========================================

st.subheader("⚠️ Credit Risk Distribution")

fig = px.histogram(
    df,
    x="credit_risk_score",
    nbins=50
)

fig.update_traces(
    marker_color="#ef4444"
)

fig.update_layout(
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==========================================
# FRAUD VS TRUST SCORE
# ==========================================

st.subheader("🛡 Trust Score vs Fraud")

sample_df = df.sample(
    min(5000,len(df)),
    random_state=42
)

fig = px.scatter(
    sample_df,
    x="customer_trust_score",
    y="credit_risk_score",
    color="fraud_bool"
)

fig.update_layout(
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==========================================
# DEVICE FRAUD ANALYSIS
# ==========================================

st.subheader("📱 Device Fraud Analysis")

fig = px.box(
    sample_df,
    x="fraud_bool",
    y="device_fraud_count",
    color="fraud_bool"
)

fig.update_layout(
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==========================================
# TRANSACTION VELOCITY RISK
# ==========================================

st.subheader("⚡ Transaction Velocity Intelligence")

fig = px.scatter(
    sample_df,
    x="transaction_velocity_score",
    y="device_fraud_count",
    color="fraud_bool"
)

fig.update_layout(
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==========================================
# TOP SUSPICIOUS RECORDS
# ==========================================

st.subheader("🚨 Top Suspicious Transactions")

suspicious = df.sort_values(
    [
        "credit_risk_score",
        "device_fraud_count"
    ],
    ascending=False
).head(25)

st.dataframe(
    suspicious[
        [
            "credit_risk_score",
            "customer_trust_score",
            "transaction_velocity_score",
            "device_fraud_count",
            "foreign_request",
            "fraud_bool"
        ]
    ],
    width="stretch"
)

# ==========================================
# FRAUD INDICATOR SCORES
# ==========================================

st.subheader("🎯 Fraud Indicator Analysis")

indicator_cols = [

    "credit_risk_score",

    "customer_trust_score",

    "transaction_velocity_score",

    "digital_footprint",

    "device_fraud_count"

]

indicator_df = pd.DataFrame({
    "Indicator": indicator_cols,
    "Average Score": [
        df[col].mean()
        for col in indicator_cols
    ]
})

fig = px.bar(
    indicator_df,
    x="Indicator",
    y="Average Score"
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

# ==========================================
# FRAUD PROFILE SUMMARY
# ==========================================

st.subheader("📋 Fraud Intelligence Summary")

st.success("""
✔ High Risk Customer Detection

✔ Foreign Transaction Intelligence

✔ Device Fraud Monitoring

✔ Velocity Pattern Analysis

✔ Suspicious Transaction Identification

✔ Fraud Indicator Tracking

✔ Trust Score Investigation
""")