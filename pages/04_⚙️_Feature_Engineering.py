import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Feature Engineering",
    page_icon="⚙️",
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
<h1>⚙️ Feature Engineering Dashboard</h1>
<h3>Advanced Fraud Intelligence Features</h3>
<p>
Explore engineered variables used to improve fraud detection performance.
</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# FEATURE LIST
# ==========================================

engineered_features = [

    "address_stability",

    "velocity_ratio",

    "phone_verification_score",

    "bank_relationship",

    "customer_trust_score",

    "transaction_velocity_score",

    "digital_footprint"

]

# ==========================================
# KPI SECTION
# ==========================================

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.metric(
        "Engineered Features",
        len(engineered_features)
    )

with c2:
    st.metric(
        "Dataset Rows",
        f"{len(df):,}"
    )

with c3:
    st.metric(
        "Fraud Cases",
        int(df["fraud_bool"].sum())
    )

with c4:
    st.metric(
        "Feature Groups",
        4
    )

# ==========================================
# FEATURE DISTRIBUTIONS
# ==========================================

st.subheader("📊 Engineered Feature Distribution")

selected_feature = st.selectbox(
    "Select Feature",
    engineered_features
)

fig = px.histogram(
    df,
    x=selected_feature,
    nbins=40,
    title=f"{selected_feature} Distribution"
)

fig.update_layout(
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==========================================
# FRAUD IMPACT ANALYSIS
# ==========================================

st.subheader("🚨 Feature Impact on Fraud")

fig = px.box(
    df.sample(
        min(10000, len(df)),
        random_state=42
    ),
    x="fraud_bool",
    y=selected_feature,
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
# TRUST SCORE ANALYSIS
# ==========================================

st.subheader("🛡 Customer Trust Score")

fig = px.histogram(
    df,
    x="customer_trust_score",
    nbins=50,
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

# ==========================================
# DIGITAL FOOTPRINT
# ==========================================

st.subheader("🌐 Digital Footprint Analysis")

fig = px.histogram(
    df,
    x="digital_footprint",
    nbins=40,
    title="Digital Footprint Distribution"
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

# ==========================================
# VELOCITY SCORE
# ==========================================

st.subheader("⚡ Transaction Velocity Score")

fig = px.scatter(
    df.sample(
        min(5000, len(df)),
        random_state=42
    ),
    x="transaction_velocity_score",
    y="credit_risk_score",
    color="fraud_bool",
    title="Velocity Score vs Credit Risk"
)

fig.update_layout(
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==========================================
# CORRELATION ANALYSIS
# ==========================================

st.subheader("🔗 Engineered Feature Correlation")

corr = (
    df[
        engineered_features
    ]
    .corr()
)

fig = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="Blues"
)

fig.update_layout(
    height=700,
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==========================================
# FEATURE IMPORTANCE TABLE
# ==========================================

st.subheader("📋 Engineered Features Summary")

summary = df[
    engineered_features
].describe().T

st.dataframe(
    summary,
    width="stretch"
)

# ==========================================
# SUMMARY
# ==========================================

st.markdown("---")

st.success("""
Feature Engineering Completed

✔ Address Stability Score

✔ Velocity Ratio

✔ Phone Verification Score

✔ Bank Relationship Score

✔ Customer Trust Score

✔ Transaction Velocity Score

✔ Digital Footprint Score

These engineered features significantly improve fraud detection capability.
""")