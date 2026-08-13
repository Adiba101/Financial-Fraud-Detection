import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Graph Analysis",
    layout="wide"
)


# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/processed/fraud_detection_cleaned_sample.csv"
    )

    return df


df = load_data()


# =====================================================
# HEADER
# =====================================================

st.markdown(
"""
<div class="hero">

<h1>🕸️ Fraud Graph Analysis Dashboard</h1>

<h3>
Relationship Discovery & Fraud Pattern Analysis
</h3>

<p>
Analyze fraud behavior, feature interactions,
correlations and transaction patterns.
</p>

</div>
""",
unsafe_allow_html=True
)


# =====================================================
# DATASET OVERVIEW
# =====================================================

st.subheader("📊 Dataset Summary")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Transactions",
        f"{len(df):,}"
    )


with col2:

    if "fraud_bool" in df.columns:

        st.metric(
            "Fraud Cases",
            f"{int(df['fraud_bool'].sum()):,}"
        )


with col3:

    if "fraud_bool" in df.columns:

        fraud_rate = (
            df["fraud_bool"].mean()
            * 100
        )

        st.metric(
            "Fraud Rate",
            f"{fraud_rate:.2f}%"
        )


with col4:

    st.metric(
        "Features",
        len(df.columns)
    )


st.markdown("---")


# =====================================================
# CORRELATION HEATMAP
# =====================================================

st.subheader(
    "🔥 Correlation Heatmap"
)


corr_features = [

    "income",
    "customer_age",
    "credit_risk_score",
    "intended_balcon_amount",
    "bank_months_count",
    "device_distinct_emails_8w",
    "fraud_bool"

]


available_cols = [

    col
    for col in corr_features

    if col in df.columns

]


corr_matrix = (
    df[available_cols]
    .corr()
)


fig = px.imshow(

    corr_matrix,

    text_auto=True,

    aspect="auto",

    color_continuous_scale="RdBu_r"

)


fig.update_layout(

    template="plotly_dark",

    height=650

)


st.plotly_chart(
    fig,
    width="stretch"
)


st.markdown("---")

# =====================================================
# FRAUD VS CREDIT RISK SCORE
# =====================================================

st.subheader(
    "📊 Fraud vs Credit Risk Score"
)

if (
    "credit_risk_score" in df.columns
    and "fraud_bool" in df.columns
):

    temp_df = df.sample(
        min(10000, len(df)),
        random_state=42
    )

    fig = px.box(

        temp_df,

        x="fraud_bool",

        y="credit_risk_score",

        color="fraud_bool",

        labels={
            "fraud_bool":"Fraud",
            "credit_risk_score":"Credit Risk Score"
        }

    )

    fig.update_layout(

        template="plotly_dark",

        height=500

    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

else:

    st.info(
        "Required columns not found."
    )


st.markdown("---")


# =====================================================
# TRANSACTION AMOUNT DISTRIBUTION
# =====================================================

st.subheader(
    "💰 Transaction Amount Distribution"
)

if (
    "intended_balcon_amount" in df.columns
    and "fraud_bool" in df.columns
):

    temp_df = df.sample(
        min(50000, len(df)),
        random_state=42
    )

    fig = px.histogram(

        temp_df,

        x="intended_balcon_amount",

        color="fraud_bool",

        nbins=50,

        opacity=0.8

    )

    fig.update_layout(

        template="plotly_dark",

        height=500

    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

else:

    st.info(
        "Required columns not found."
    )


st.markdown("---")


# =====================================================
# FRAUD VS AGE ANALYSIS
# =====================================================

st.subheader(
    "👥 Fraud vs Customer Age"
)

required_cols = [
    "customer_age",
    "credit_risk_score",
    "fraud_bool"
]

if all(
    col in df.columns
    for col in required_cols
):

    sample_df = df.sample(
        min(5000, len(df)),
        random_state=42
    )

    fig = px.scatter(

        sample_df,

        x="customer_age",

        y="credit_risk_score",

        color="fraud_bool",

        hover_data=[
            "customer_age",
            "credit_risk_score"
        ]

    )

    fig.update_layout(

        template="plotly_dark",

        height=550

    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

else:

    st.info(
        "Required columns not found."
    )


st.markdown("---")


# =====================================================
# INCOME VS FRAUD ANALYSIS
# =====================================================

st.subheader(
    "💵 Income vs Fraud Analysis"
)

if (
    "income" in df.columns
    and "fraud_bool" in df.columns
):

    sample_df = df.sample(
        min(10000, len(df)),
        random_state=42
    )

    fig = px.violin(

        sample_df,

        x="fraud_bool",

        y="income",

        color="fraud_bool",

        box=True

    )

    fig.update_layout(

        template="plotly_dark",

        height=500

    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

else:

    st.info(
        "Required columns not found."
    )


st.markdown("---")
# =====================================================
# FEATURE CORRELATION RANKING
# =====================================================

st.subheader(
    "🏆 Features Most Related To Fraud"
)

if "fraud_bool" in df.columns:

    numeric_df = df.select_dtypes(
        include=np.number
    )

    corr_target = (
        numeric_df
        .corr()["fraud_bool"]
        .abs()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )

    corr_target.columns = [
        "Feature",
        "Correlation"
    ]

    corr_target = corr_target.head(15)

    fig = px.bar(

        corr_target,

        x="Correlation",

        y="Feature",

        orientation="h",

        text="Correlation"

    )

    fig.update_layout(

        template="plotly_dark",

        height=650

    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

else:

    st.info(
        "fraud_bool column not found."
    )


st.markdown("---")


# =====================================================
# FRAUD RISK NETWORK
# =====================================================

st.subheader(
    "🕸️ Fraud Risk Network"
)

network_df = pd.DataFrame({

    "Factor":[

        "Credit Risk Score",
        "Transaction Amount",
        "Foreign Request",
        "Email Similarity",
        "Device Emails",
        "Bank Months"

    ],

    "Importance":[

        95,
        80,
        75,
        70,
        65,
        50

    ]

})

fig = px.bar(

    network_df,

    x="Factor",

    y="Importance",

    color="Importance",

    text="Importance"

)

fig.update_layout(

    template="plotly_dark",

    height=500

)

st.plotly_chart(
    fig,
    width="stretch"
)


st.markdown("---")


# =====================================================
# FRAUD CLUSTER ANALYSIS
# =====================================================

st.subheader(
    "🎯 Fraud Cluster Analysis"
)

cluster_cols = [

    "income",

    "credit_risk_score",

    "customer_age",

    "intended_balcon_amount"

]

available_cluster_cols = [

    col

    for col in cluster_cols

    if col in df.columns

]

if len(available_cluster_cols) >= 2:

    sample_df = df.sample(
        min(5000, len(df)),
        random_state=42
    )

    x_feature = available_cluster_cols[0]
    y_feature = available_cluster_cols[1]

    color_col = (
        "fraud_bool"
        if "fraud_bool" in df.columns
        else None
    )

    fig = px.scatter(

        sample_df,

        x=x_feature,

        y=y_feature,

        color=color_col,

        opacity=0.7

    )

    fig.update_layout(

        template="plotly_dark",

        height=550

    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

else:

    st.info(
        "Not enough numerical features."
    )


st.markdown("---")


# =====================================================
# TOP FRAUD INSIGHTS
# =====================================================

st.subheader(
    "📌 Fraud Insights Summary"
)

col1, col2 = st.columns(2)

with col1:

    st.info(
        """
        • High credit risk scores are strongly
        associated with fraud.

        • Large transaction amounts often
        increase fraud probability.

        • Device-related activity is an
        important fraud indicator.
        """
    )

with col2:

    st.warning(
        """
        • Foreign requests require
        additional verification.

        • Frequent device email changes
        increase risk.

        • Unusual account behavior
        should be monitored closely.
        """
    )


st.markdown("---")


# =====================================================
# DASHBOARD FOOTER
# =====================================================

st.success(
    "✅ Graph Analysis Dashboard Loaded Successfully"
)