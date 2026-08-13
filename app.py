import streamlit as st

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
with open("streamlit_app/assets/style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# Sidebar Branding
st.sidebar.markdown(
"""
<div class="sidebar-logo">

<h1>🛡️</h1>

<h2>FraudShield AI</h2>

<p>
Enterprise Fraud Detection Platform
</p>

</div>
""",
unsafe_allow_html=True
)

st.sidebar.markdown("---")

# HERO
st.markdown(
"""
<div class="hero">

<h1>🛡️ FraudShield AI</h1>

<h3>
Enterprise Fraud Detection &
Risk Intelligence Platform
</h3>

<p>
Real-Time Monitoring • Machine Learning •
Risk Scoring • Fraud Investigation • Graph Analytics
</p>

</div>
""",
unsafe_allow_html=True
)

# KPI ROW
c1,c2,c3,c4 = st.columns(4)

with c1:
    st.metric(
        "Transactions",
        "1,000,000"
    )

with c2:
    st.metric(
        "ML Models",
        "6"
    )

with c3:
    st.metric(
        "Notebooks",
        "19"
    )

with c4:
    st.metric(
        "Best Model",
        "XGBoost"
    )

st.markdown("<br>", unsafe_allow_html=True)

# PROJECT OVERVIEW
st.markdown(
"""
<div class="glass">

<h2>🚀 Project Overview</h2>

<p>

This platform detects fraudulent financial
transactions using Machine Learning,
Anomaly Detection, Graph Analytics,
Risk Scoring and Real-Time Monitoring.

The system evaluates customer behavior,
transaction velocity, device activity,
credit risk indicators and fraud patterns.

</p>

</div>
""",
unsafe_allow_html=True
)

# WORKFLOW
st.markdown(
"""
<div class="section-title">
⚙️ AI Fraud Detection Pipeline
</div>
""",
unsafe_allow_html=True
)

col1,col2,col3,col4,col5,col6,col7 = st.columns(7)

steps = [
"📥 Data",
"🧹 Clean",
"📊 EDA",
"⚖️ SMOTE",
"⚙️ Feature",
"🤖 Models",
"🚨 Detect"
]

for col,step in zip(
    [col1,col2,col3,col4,col5,col6,col7],
    steps
):
    with col:
        st.markdown(
        f"""
        <div class="workflow-card">
        <h4>{step}</h4>
        </div>
        """,
        unsafe_allow_html=True
        )

st.markdown("<br>", unsafe_allow_html=True)

# FEATURES
st.markdown(
"""
<div class="section-title">
🌟 Platform Features
</div>
""",
unsafe_allow_html=True
)

a,b,c = st.columns(3)

with a:
    st.success("🤖 Machine Learning Models")
    st.success("🎯 Fraud Risk Scoring")
    st.success("📊 Executive Dashboard")

with b:
    st.info("📡 Real-Time Monitoring")
    st.info("🕵️ Fraud Intelligence")
    st.info("🔍 AI Investigation")

with c:
    st.warning("📈 Fraud Forecasting")
    st.warning("🕸️ Graph Analytics")
    st.warning("📁 Dataset Explorer")

st.markdown("---")

st.caption(
    "FraudShield AI • Financial Fraud Detection Platform "
    "Developed by ADIBA ANSARI "
)