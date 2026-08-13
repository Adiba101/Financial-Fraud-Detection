import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load("models/xgboost.pkl")


model = load_model()


# =====================================================
# LOAD FEATURE IMPORTANCE
# =====================================================

@st.cache_data
def load_importance():

    return pd.read_csv(
        "data/processed/xgboost_feature_importance.csv"
    )


importance_df = load_importance()



# =====================================================
# PAGE HEADER
# =====================================================

st.markdown(
"""
<div class="hero">

<h1>🚨 AI Fraud Prediction Center</h1>

<h3>
Real-Time Financial Fraud Detection System
</h3>

<p>
Predict fraudulent transactions using Machine Learning and XGBoost.
</p>

</div>
""",
unsafe_allow_html=True
)



# =====================================================
# INPUT FORM
# =====================================================


st.markdown(
"<div class='section-title'>📝 Transaction Information</div>",
unsafe_allow_html=True
)



col1, col2 = st.columns(2)



# ---------------- LEFT COLUMN ----------------

with col1:


    income = st.number_input(
        "Income",
        value=5000.0
    )


    customer_age = st.slider(
        "Customer Age",
        18,
        90,
        35
    )


    credit_risk_score = st.slider(
        "Credit Risk Score",
        0,
        500,
        100
    )


    bank_months_count = st.slider(
        "Bank Relationship Months",
        0,
        60,
        12
    )


    current_address_months_count = st.slider(
        "Current Address Months",
        0,
        300,
        24
    )


    proposed_credit_limit = st.number_input(
        "Proposed Credit Limit",
        value=1500.0
    )



# ---------------- RIGHT COLUMN ----------------


with col2:


    device_distinct_emails_8w = st.slider(
        "Device Distinct Emails (8 Weeks)",
        0,
        20,
        1
    )


    days_since_request = st.number_input(
        "Days Since Request",
        value=1.0
    )


    intended_balcon_amount = st.number_input(
        "Transaction Amount",
        value=500.0
    )


    name_email_similarity = st.slider(
        "Name Email Similarity",
        0.0,
        1.0,
        0.5
    )


    session_length_in_minutes = st.slider(
        "Session Length",
        0.0,
        60.0,
        5.0
    )




# =====================================================
# SECURITY DETAILS
# =====================================================


st.markdown(
"<div class='section-title'>🔒 Security Information</div>",
unsafe_allow_html=True
)



c1, c2, c3 = st.columns(3)



with c1:


    phone_home_valid = st.selectbox(
        "Phone Home Valid",
        [0,1]
    )


    phone_mobile_valid = st.selectbox(
        "Phone Mobile Valid",
        [0,1]
    )



with c2:


    keep_alive_session = st.selectbox(
        "Keep Alive Session",
        [0,1]
    )


    has_other_cards = st.selectbox(
        "Has Other Cards",
        [0,1]
    )



with c3:


    foreign_request = st.selectbox(
        "Foreign Request",
        [0,1]
    )


    email_is_free = st.selectbox(
        "Email Is Free",
        [0,1]
    )



# =====================================================
# TRANSACTION PROFILE
# =====================================================


st.markdown(
"<div class='section-title'>⚙️ Transaction Profile</div>",
unsafe_allow_html=True
)



a,b,c,d = st.columns(4)



with a:

    payment_type = st.selectbox(
        "Payment Type",
        [1,2,3,4]
    )


with b:

    employment_status = st.selectbox(
        "Employment Status",
        [1,2,3,4,5,6]
    )


with c:

    housing_status = st.selectbox(
        "Housing Status",
        [1,2,3,4,5,6]
    )


with d:

    device_os = st.selectbox(
        "Device OS",
        [1,2,3,4]
    )



# =====================================================
# PREDICT BUTTON START
# =====================================================


if st.button("🚀 Predict Fraud Risk"):


    features = [

        'income',
        'name_email_similarity',
        'prev_address_months_count',
        'current_address_months_count',
        'customer_age',
        'days_since_request',
        'intended_balcon_amount',
        'zip_count_4w',
        'velocity_6h',
        'velocity_24h',
        'velocity_4w',
        'bank_branch_count_8w',
        'date_of_birth_distinct_emails_4w',
        'credit_risk_score',
        'email_is_free',
        'phone_home_valid',
        'phone_mobile_valid',
        'bank_months_count',
        'has_other_cards',
        'proposed_credit_limit',
        'foreign_request',
        'session_length_in_minutes',
        'keep_alive_session',
        'device_distinct_emails_8w',
        'device_fraud_count',
        'month',

        'payment_type_1',
        'payment_type_2',
        'payment_type_3',
        'payment_type_4',

        'employment_status_1',
        'employment_status_2',
        'employment_status_3',
        'employment_status_4',
        'employment_status_5',
        'employment_status_6',

        'housing_status_1',
        'housing_status_2',
        'housing_status_3',
        'housing_status_4',
        'housing_status_5',
        'housing_status_6',

        'source_1',

        'device_os_1',
        'device_os_2',
        'device_os_3',
        'device_os_4'

    ]

    # =====================================================
    # CREATE INPUT DATAFRAME
    # =====================================================


    input_df = pd.DataFrame(
        np.zeros((1, len(features))),
        columns=features
    )


    # Numerical features

    input_df["income"] = income

    input_df["customer_age"] = customer_age

    input_df["credit_risk_score"] = credit_risk_score

    input_df["bank_months_count"] = bank_months_count

    input_df["proposed_credit_limit"] = proposed_credit_limit

    input_df["current_address_months_count"] = current_address_months_count

    input_df["device_distinct_emails_8w"] = device_distinct_emails_8w

    input_df["days_since_request"] = days_since_request

    input_df["intended_balcon_amount"] = intended_balcon_amount

    input_df["name_email_similarity"] = name_email_similarity

    input_df["session_length_in_minutes"] = session_length_in_minutes



    # Security features

    input_df["phone_home_valid"] = phone_home_valid

    input_df["phone_mobile_valid"] = phone_mobile_valid

    input_df["keep_alive_session"] = keep_alive_session

    input_df["has_other_cards"] = has_other_cards

    input_df["foreign_request"] = foreign_request

    input_df["email_is_free"] = email_is_free



    # One hot encoding

    input_df[f"payment_type_{payment_type}"] = 1

    input_df[f"employment_status_{employment_status}"] = 1

    input_df[f"housing_status_{housing_status}"] = 1

    input_df[f"device_os_{device_os}"] = 1



    # =====================================================
    # MODEL PREDICTION
    # =====================================================


    prediction = model.predict(input_df)[0]


    try:

        probability = model.predict_proba(input_df)[0][1]

    except:

        probability = 0.50



    st.markdown("---")



    # =====================================================
    # RESULT
    # =====================================================


    if prediction == 1:

        st.error(
            f"🚨 FRAUD DETECTED | Probability: {probability:.2%}"
        )


    else:

        st.success(
            f"✅ LEGITIMATE TRANSACTION | Probability: {probability:.2%}"
        )




    # =====================================================
    # FRAUD RISK GAUGE
    # =====================================================


    st.subheader("🎯 Fraud Risk Gauge")



    gauge = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=probability*100,

            title={
                "text":"Fraud Risk %"
            },


            gauge={

                "axis":{
                    "range":[0,100]
                },

                "bar":{
                    "color":"crimson"
                }

            }

        )

    )



    gauge.update_layout(
        height=400
    )


    st.plotly_chart(
        gauge,
        width="stretch"
    )




    # =====================================================
    # SAFE VS FRAUD DONUT + USER METRICS
    # =====================================================


    col1, col2 = st.columns(2)



    with col1:


        pie_df = pd.DataFrame({

            "Category":[
                "Safe",
                "Fraud"
            ],

            "Value":[

                (1-probability)*100,

                probability*100

            ]

        })



        fig = px.pie(

            pie_df,

            names="Category",

            values="Value",

            hole=0.65

        )


        fig.update_layout(

            template="plotly_dark"

        )


        st.plotly_chart(

            fig,

            width="stretch"

        )




    with col2:


        risk_df = pd.DataFrame({

            "Metric":[

                "Income",

                "Credit Risk",

                "Age",

                "Bank Months"

            ],


            "Value":[

                income,

                credit_risk_score,

                customer_age,

                bank_months_count

            ]

        })



        fig = px.bar(

            risk_df,

            x="Metric",

            y="Value"

        )



        fig.update_layout(

            template="plotly_dark"

        )


        st.plotly_chart(

            fig,

            width="stretch"

        )




    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================


    st.subheader("🔥 Top Fraud Indicators")



    top15 = importance_df.head(15)



    fig = px.bar(

        top15,

        x="Importance",

        y="Feature",

        orientation="h"

    )


    fig.update_layout(

        template="plotly_dark",

        height=600

    )



    st.plotly_chart(

        fig,

        width="stretch"

    )
    # =====================================================
    # AI RECOMMENDATION
    # =====================================================


    st.subheader("🤖 AI Recommendation")



    if probability < 0.25:

        st.success(
            "🟢 Low Risk Transaction"
        )


    elif probability < 0.50:

        st.info(
            "🔵 Moderate Risk Transaction"
        )


    elif probability < 0.75:

        st.warning(
            "🟠 High Risk Transaction"
        )


    else:

        st.error(
            "🔴 Critical Fraud Risk - Immediate Review Required"
        )



    st.markdown("---")




    # =====================================================
    # CUSTOMER RISK PROFILE
    # =====================================================


    st.subheader("🎯 Customer Risk Profile")



    radar_df = pd.DataFrame({

        "Feature":[

            "Credit Risk",

            "Income",

            "Age",

            "Bank Months",

            "Address Stability",

            "Device Trust"

        ],


        "Value":[

            credit_risk_score,

            income/100,

            customer_age,

            bank_months_count,

            current_address_months_count,

            device_distinct_emails_8w*10

        ]

    })




    radar_fig = go.Figure()



    radar_fig.add_trace(

        go.Scatterpolar(

            r=radar_df["Value"],

            theta=radar_df["Feature"],

            fill="toself",

            name="Customer Profile"

        )

    )



    radar_fig.update_layout(

        template="plotly_dark",

        height=500,

        polar=dict(

            radialaxis=dict(

                visible=True

            )

        )

    )



    st.plotly_chart(

        radar_fig,

        width="stretch"

    )





    # =====================================================
    # TRANSACTION ANALYSIS
    # =====================================================


    st.subheader("📊 Transaction Analysis")



    feature_df = pd.DataFrame({

        "Feature":[

            "Income",

            "Credit Risk",

            "Bank Months",

            "Credit Limit",

            "Transaction Amount"

        ],


        "Value":[

            income,

            credit_risk_score,

            bank_months_count,

            proposed_credit_limit,

            intended_balcon_amount

        ]

    })




    fig = px.bar(

        feature_df,

        x="Feature",

        y="Value",

        text="Value"

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
    # FRAUD RISK TREND SIMULATION
    # =====================================================


    st.subheader("📈 Fraud Risk Trend Simulation")



    trend_df = pd.DataFrame({

        "Step":[

            1,

            2,

            3,

            4,

            5

        ],


        "Risk":[

            probability*40,

            probability*60,

            probability*75,

            probability*90,

            probability*100

        ]

    })



    trend_fig = px.line(

        trend_df,

        x="Step",

        y="Risk",

        markers=True

    )



    trend_fig.update_layout(

        template="plotly_dark",

        height=400

    )



    st.plotly_chart(

        trend_fig,

        width="stretch"

    )


# =====================================================
# END OF FILE
# =====================================================