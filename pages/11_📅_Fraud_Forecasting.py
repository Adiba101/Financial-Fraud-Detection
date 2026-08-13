import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Fraud Forecasting",
    layout="wide"
)

st.title("📈 Fraud Forecasting Dashboard")
st.markdown(
    "Predict future fraud activity using historical fraud trends."
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
# CHECK REQUIRED COLUMNS
# =====================================================

if "month" not in df.columns:

    st.error(
        "Month column not found in dataset."
    )

    st.stop()


if "fraud_bool" not in df.columns:

    st.error(
        "fraud_bool column not found."
    )

    st.stop()


# =====================================================
# HISTORICAL FRAUD TREND
# =====================================================

monthly_fraud = (
    df.groupby("month")["fraud_bool"]
    .sum()
    .reset_index()
)

monthly_fraud.columns = [
    "Month",
    "Fraud_Count"
]


st.subheader("📊 Historical Fraud Trend")

fig = px.line(
    monthly_fraud,
    x="Month",
    y="Fraud_Count",
    markers=True
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
# FORECAST MODEL
# =====================================================

X = monthly_fraud[["Month"]]

y = monthly_fraud["Fraud_Count"]

model = LinearRegression()

model.fit(X, y)


# =====================================================
# FUTURE MONTHS
# =====================================================

last_month = int(
    monthly_fraud["Month"].max()
)

future_months = pd.DataFrame({

    "Month": np.arange(
        last_month + 1,
        last_month + 7
    )

})


future_months["Forecast_Fraud"] = (
    model.predict(
        future_months[["Month"]]
    )
)

future_months["Forecast_Fraud"] = (
    future_months["Forecast_Fraud"]
    .round()
    .astype(int)
)


# =====================================================
# FORECAST TABLE
# =====================================================

st.subheader("🔮 Next 6-Month Fraud Forecast")

st.dataframe(
    future_months,
    use_container_width=True
)


# =====================================================
# COMBINED TREND + FORECAST
# =====================================================

history = monthly_fraud.copy()

history["Type"] = "Historical"

history.rename(
    columns={
        "Fraud_Count": "Value"
    },
    inplace=True
)

forecast = future_months.copy()

forecast["Type"] = "Forecast"

forecast.rename(
    columns={
        "Forecast_Fraud": "Value"
    },
    inplace=True
)

combined = pd.concat(
    [history, forecast],
    ignore_index=True
)


st.subheader("📈 Historical vs Forecast")

fig = px.line(
    combined,
    x="Month",
    y="Value",
    color="Type",
    markers=True
)

fig.update_layout(
    template="plotly_dark",
    height=500
)

st.plotly_chart(
    fig,
    width="stretch"
)


# =====================================================
# FORECAST KPIs
# =====================================================

st.subheader("📌 Forecast Summary")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Current Fraud Cases",
        int(
            monthly_fraud[
                "Fraud_Count"
            ].iloc[-1]
        )
    )

with col2:

    st.metric(
        "Next Month Forecast",
        int(
            future_months[
                "Forecast_Fraud"
            ].iloc[0]
        )
    )

with col3:

    growth = (
        (
            future_months[
                "Forecast_Fraud"
            ].iloc[0]
            -
            monthly_fraud[
                "Fraud_Count"
            ].iloc[-1]
        )
        /
        monthly_fraud[
            "Fraud_Count"
        ].iloc[-1]
    ) * 100

    st.metric(
        "Forecast Growth %",
        f"{growth:.2f}%"
    )


# =====================================================
# RISK LEVEL
# =====================================================

avg_future = (
    future_months[
        "Forecast_Fraud"
    ]
    .mean()
)

st.subheader("🚨 Forecast Risk Assessment")

if avg_future < 100:

    st.success(
        "🟢 Low Future Fraud Risk"
    )

elif avg_future < 500:

    st.warning(
        "🟠 Moderate Future Fraud Risk"
    )

else:

    st.error(
        "🔴 High Future Fraud Risk"
    )


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.success(
    "✅ Fraud Forecasting Dashboard Loaded Successfully"
)