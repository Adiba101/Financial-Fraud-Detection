import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime


st.set_page_config(
    page_title="Real-Time Monitoring",
    layout="wide"
)

st.title("📡 Real-Time Fraud Monitoring Center")
st.markdown("Monitor live transaction activity and fraud alerts.")


# =====================================================
# GENERATE LIVE DATA
# =====================================================

@st.cache_data(ttl=5)
def generate_transactions():

    n = 100

    df = pd.DataFrame({

        "Transaction_ID": [
            f"TXN{i:05d}" for i in range(n)
        ],

        "Amount": np.random.randint(
            100,
            50000,
            n
        ),

        "Risk_Score": np.random.randint(
            0,
            100,
            n
        ),

        "Timestamp": [
            datetime.now()
            for _ in range(n)
        ]

    })

    df["Status"] = np.where(
        df["Risk_Score"] >= 70,
        "Fraud",
        "Legitimate"
    )

    return df


df = generate_transactions()


# =====================================================
# KPI SECTION
# =====================================================

total_txn = len(df)

fraud_txn = len(
    df[df["Status"] == "Fraud"]
)

fraud_rate = (
    fraud_txn / total_txn
) * 100


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Transactions",
        total_txn
    )

with col2:
    st.metric(
        "Fraud Alerts",
        fraud_txn
    )

with col3:
    st.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )


st.markdown("---")


# =====================================================
# LIVE ALERTS
# =====================================================

st.subheader("🚨 High Risk Transactions")

alerts = df[
    df["Risk_Score"] >= 70
].sort_values(
    "Risk_Score",
    ascending=False
)

st.dataframe(
    alerts.head(20),
    use_container_width=True
)


# =====================================================
# RISK DISTRIBUTION
# =====================================================

st.subheader("📊 Risk Score Distribution")

fig = px.histogram(
    df,
    x="Risk_Score",
    nbins=20
)

fig.update_layout(
    template="plotly_dark",
    height=400
)

st.plotly_chart(
    fig,
    width="stretch"
)


# =====================================================
# STATUS DISTRIBUTION
# =====================================================

st.subheader("📈 Fraud vs Legitimate")

status_counts = (
    df["Status"]
    .value_counts()
    .reset_index()
)

status_counts.columns = [
    "Status",
    "Count"
]

fig = px.pie(
    status_counts,
    names="Status",
    values="Count",
    hole=0.6
)

fig.update_layout(
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    width="stretch"
)


# =====================================================
# LIVE TRANSACTION FEED
# =====================================================

st.subheader("📋 Live Transaction Feed")

feed = df.sort_values(
    "Timestamp",
    ascending=False
)

st.dataframe(
    feed,
    use_container_width=True
)


# =====================================================
# REFRESH
# =====================================================

st.button(
    "🔄 Refresh Monitoring Data"
)