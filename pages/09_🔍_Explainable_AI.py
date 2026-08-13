import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    return joblib.load(
        "models/xgboost.pkl"
    )


model = load_model()



# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "data/processed/fraud_detection_cleaned_sample.csv"
    )


df = load_data()



# =====================================================
# HEADER
# =====================================================


st.markdown(
"""
<div class="hero">

<h1>🔍 Explainable AI Dashboard</h1>

<h3>
Understanding Why AI Detects Fraud
</h3>

<p>
SHAP based model interpretation for transparent fraud detection.
</p>

</div>
""",
unsafe_allow_html=True
)




# =====================================================
# PREPARE FEATURES SAME AS MODEL TRAINING
# =====================================================

X = df.drop(
    "fraud_bool",
    axis=1
)


# Keep only columns used by model

model_features = model.get_booster().feature_names


X = X.reindex(
    columns=model_features,
    fill_value=0
)



# =====================================================
# SHAP EXPLAINER
# =====================================================


@st.cache_resource
def create_explainer():

    return shap.TreeExplainer(
        model
    )


explainer = create_explainer()



# =====================================================
# GLOBAL FEATURE IMPORTANCE
# =====================================================


st.subheader(
    "🔥 Global Fraud Risk Factors"
)


sample = X.sample(
    min(1000,len(X)),
    random_state=42
)



shap_values = explainer(
    sample
).values


fig, ax = plt.subplots(
    figsize=(10,6)
)


shap.summary_plot(

    shap_values,

    sample,

    show=False

)



st.pyplot(
    fig
)




# =====================================================
# FEATURE IMPORTANCE TABLE
# =====================================================


st.subheader(
    "📊 Feature Contribution Ranking"
)



importance = pd.DataFrame({

    "Feature":
    sample.columns,


    "Importance":
    np.abs(shap_values).mean(axis=0)

})


importance = importance.sort_values(

    "Importance",

    ascending=False

)



st.dataframe(

    importance.head(20),

    use_container_width=True

)





# =====================================================
# SINGLE TRANSACTION EXPLANATION
# =====================================================


st.markdown("---")


st.subheader(
    "🚨 Explain Individual Transaction"
)



index = st.number_input(

    "Select Transaction Index",

    min_value=0,

    max_value=len(X)-1,

    value=0

)



transaction = X.iloc[
    [index]
]




prediction = model.predict(
    transaction
)[0]



probability = model.predict_proba(
    transaction
)[0][1]




if prediction == 1:

    st.error(

        f"🚨 Fraud Detected | Risk: {probability:.2%}"

    )

else:

    st.success(

        f"✅ Legitimate Transaction | Risk: {probability:.2%}"

    )





# =====================================================
# LOCAL SHAP EXPLANATION
# =====================================================


st.subheader(
    "🧠 Why This Prediction?"
)



local_shap = explainer(
    transaction
).values


local_df = pd.DataFrame({

    "Feature":
    transaction.columns,


    "Value":
    transaction.iloc[0].values,


    "Impact":
    local_shap[0]

})



local_df["Absolute Impact"] = (

    local_df["Impact"]
    .abs()

)



local_df = local_df.sort_values(

    "Absolute Impact",

    ascending=False

)



st.dataframe(

    local_df.head(15),

    use_container_width=True

)




# =====================================================
# WATERFALL CHART
# =====================================================


st.subheader(
    "📈 Fraud Decision Explanation"
)



fig = plt.figure(
    figsize=(10,6)
)



shap.plots.waterfall(

    shap.Explanation(

        values=local_shap[0],

        base_values=explainer.expected_value,

        data=transaction.iloc[0],

        feature_names=transaction.columns

    ),

    show=False

)



st.pyplot(
    plt.gcf()
)





# =====================================================
# TOP FRAUD DRIVERS
# =====================================================


st.markdown("---")


st.subheader(
    "⚠️ Top Reasons Increasing Fraud Risk"
)



positive = local_df[
    local_df["Impact"] > 0
].head(5)



if len(positive) > 0:

    for _,row in positive.iterrows():

        st.warning(

            f"""
            **{row['Feature']}**

            Increased fraud probability

            Impact: {row['Impact']:.4f}

            """

        )

else:

    st.success(
        "No major fraud indicators detected."
    )




# =====================================================
# FOOTER
# =====================================================


st.markdown("---")

st.success(
    "✅ Explainable AI Dashboard "
)