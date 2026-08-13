import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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
# PAGE HEADER
# =====================================================

st.markdown(
"""
<div class="hero">

<h1>📊 Fraud Analytics Dashboard</h1>

<h3>
AI Powered Financial Fraud Monitoring System
</h3>

<p>
Analyze fraud patterns, transaction behavior and risk factors.
</p>

</div>
""",
unsafe_allow_html=True
)



# =====================================================
# KPI CARDS
# =====================================================

st.markdown(
"## 📌 Fraud Overview"
)


total_transactions = len(df)


fraud_count = df["fraud_bool"].sum()


legit_count = total_transactions - fraud_count


fraud_rate = (
    fraud_count / total_transactions
) * 100



avg_amount = (
    df["intended_balcon_amount"]
    .mean()
)



col1,col2,col3,col4 = st.columns(4)



with col1:

    st.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )


with col2:

    st.metric(
        "Fraud Cases",
        f"{fraud_count:,}"
    )


with col3:

    st.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )


with col4:

    st.metric(
        "Average Amount",
        f"${avg_amount:.2f}"
    )



st.markdown("---")



# =====================================================
# FRAUD DISTRIBUTION
# =====================================================


st.subheader(
    "🔴 Fraud vs Legitimate Transactions"
)



fraud_df = pd.DataFrame({

    "Status":[
        "Legitimate",
        "Fraud"
    ],

    "Count":[
        legit_count,
        fraud_count
    ]

})



fig = px.pie(

    fraud_df,

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
# FRAUD TREND
# =====================================================


st.subheader(
    "📈 Fraud Trend Over Time"
)



if "month" in df.columns:


    trend = (
        df.groupby("month")
        ["fraud_bool"]
        .sum()
        .reset_index()
    )


    fig = px.line(

        trend,

        x="month",

        y="fraud_bool",

        markers=True,

        labels={

            "fraud_bool":"Fraud Count"

        }

    )


    fig.update_layout(

        template="plotly_dark",

        height=400

    )


    st.plotly_chart(

        fig,

        width="stretch"

    )


else:

    st.info(
        "Month column not available"
    )





# =====================================================
# PAYMENT TYPE ANALYSIS
# =====================================================


st.subheader(
    "💳 Fraud By Payment Type"
)



if "payment_type" in df.columns:


    payment = (

        df.groupby("payment_type")
        ["fraud_bool"]
        .sum()
        .reset_index()

    )


    fig = px.bar(

        payment,

        x="payment_type",

        y="fraud_bool",

        text="fraud_bool",

        labels={

            "fraud_bool":"Fraud Cases"

        }

    )


    fig.update_layout(

        template="plotly_dark"

    )


    st.plotly_chart(

        fig,

        width="stretch"

    )



else:

    st.info(
        "Payment type column not available"
    )




# =====================================================
# DEVICE ANALYSIS
# =====================================================


st.subheader(
    "📱 Fraud By Device OS"
)



if "device_os" in df.columns:


    device = (

        df.groupby("device_os")
        ["fraud_bool"]
        .sum()
        .reset_index()

    )


    fig = px.bar(

        device,

        x="device_os",

        y="fraud_bool",

        text="fraud_bool"

    )


    fig.update_layout(

        template="plotly_dark"

    )


    st.plotly_chart(

        fig,

        width="stretch"

    )


else:

    st.info(
        "Device OS column not available"
    )





# =====================================================
# RISK FACTOR ANALYSIS
# =====================================================


st.subheader(
    "🔥 Fraud Risk Factors"
)



risk_features = [

    "credit_risk_score",

    "device_fraud_count",

    "name_email_similarity",

    "velocity_24h",

    "foreign_request"

]



available_features = [

    x for x in risk_features

    if x in df.columns

]



risk_data = pd.DataFrame({

    "Feature":available_features,

    "Correlation":[

        abs(

            df[x]
            .corr(df["fraud_bool"])

        )

        for x in available_features

    ]

})



risk_data = (
    risk_data
    .sort_values(
        "Correlation",
        ascending=False
    )
)



fig = px.bar(

    risk_data,

    x="Correlation",

    y="Feature",

    orientation="h",

    text="Correlation"

)



fig.update_layout(

    template="plotly_dark",

    height=450

)



st.plotly_chart(

    fig,

    width="stretch"

)





# =====================================================
# FRAUD BY AGE GROUP
# =====================================================


st.subheader(
    "👥 Fraud Distribution By Age"
)



if "customer_age" in df.columns:


    age_df = df.copy()


    age_df["Age_Group"] = pd.cut(

        age_df["customer_age"],

        bins=[
            0,
            25,
            40,
            60,
            100
        ],

        labels=[
            "Young",
            "Adult",
            "Middle Age",
            "Senior"
        ]

    )


    age_fraud = (

        age_df.groupby(
            "Age_Group",
            observed=True
        )
        ["fraud_bool"]
        .sum()
        .reset_index()

    )


    fig = px.bar(

        age_fraud,

        x="Age_Group",

        y="fraud_bool",

        text="fraud_bool"

    )


    fig.update_layout(

        template="plotly_dark"

    )


    st.plotly_chart(

        fig,

        width="stretch"

    )



# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.success(
    "✅ Fraud Analytics Dashboard "
)