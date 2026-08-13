import streamlit as st
import pandas as pd
import plotly.express as px

# ==================================
# PAGE CONFIG
# ==================================

st.set_page_config(
    page_title="Dataset Overview",
    page_icon="📂",
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
<h1>📂 Dataset Overview</h1>
<h3>Financial Fraud Detection Dataset</h3>
<p>Explore dataset structure, quality, fraud distribution and feature information.</p>
</div>
""", unsafe_allow_html=True)

# ==================================
# KPI SECTION
# ==================================

rows = df.shape[0]
cols = df.shape[1]

fraud_cases = int(df["fraud_bool"].sum())
non_fraud = rows - fraud_cases

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.metric("📄 Rows", f"{rows:,}")

with c2:
    st.metric("📊 Columns", cols)

with c3:
    st.metric("🚨 Fraud Cases", f"{fraud_cases:,}")

with c4:
    st.metric("✅ Non-Fraud", f"{non_fraud:,}")

st.markdown("---")

# ==================================
# DATASET PREVIEW
# ==================================

st.subheader("📋 Dataset Preview")

st.dataframe(
    df.head(20),
    width="stretch"
)

# ==================================
# DATA TYPES
# ==================================

st.subheader("🔎 Data Types")

dtype_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str)
})

st.dataframe(
    dtype_df,
    width="stretch"
)

# ==================================
# MISSING VALUES
# ==================================

st.subheader("🧹 Missing Values Analysis")

missing = df.isnull().sum()

missing_df = pd.DataFrame({
    "Feature": missing.index,
    "Missing Values": missing.values
})

missing_df = missing_df[
    missing_df["Missing Values"] > 0
]

if len(missing_df) == 0:

    st.success(
        "No Missing Values Found"
    )

else:

    fig = px.bar(
        missing_df,
        x="Feature",
        y="Missing Values",
        title="Missing Values by Feature"
    )

    fig.update_layout(
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

# ==================================
# FRAUD DISTRIBUTION
# ==================================

st.subheader("🚨 Fraud Distribution")

fraud_dist = (
    df["fraud_bool"]
    .value_counts()
    .reset_index()
)

fraud_dist.columns = [
    "Fraud",
    "Count"
]

fig = px.pie(
    fraud_dist,
    names="Fraud",
    values="Count",
    hole=0.5,
    title="Fraud vs Non-Fraud"
)

fig.update_layout(
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==================================
# NUMERICAL FEATURES
# ==================================

st.subheader("📈 Numerical Features")

numeric_cols = df.select_dtypes(
    include=["int64","float64"]
).columns

selected_feature = st.selectbox(
    "Select Feature",
    numeric_cols
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

# ==================================
# CORRELATION MATRIX
# ==================================

st.subheader("🔗 Correlation Heatmap")

corr = df[numeric_cols].corr()

fig = px.imshow(
    corr,
    aspect="auto",
    color_continuous_scale="Blues"
)

fig.update_layout(
    height=800,
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ==================================
# STATISTICAL SUMMARY
# ==================================

st.subheader("📊 Statistical Summary")

st.dataframe(
    df.describe(),
    width="stretch"
)

# ==================================
# FRAUD RATE
# ==================================

st.subheader("🎯 Fraud Rate")

fraud_rate = (
    fraud_cases / rows
) * 100

st.info(
    f"Fraud Rate: {fraud_rate:.2f}%"
)

# ==================================
# FEATURE LIST
# ==================================

st.subheader("📝 Feature Inventory")

feature_df = pd.DataFrame({
    "Feature Name": df.columns
})

st.dataframe(
    feature_df,
    width="stretch"
)

# ==================================
# FOOTER
# ==================================

st.markdown("---")

st.caption(
    "FraudShield AI • Dataset Overview Dashboard"
)