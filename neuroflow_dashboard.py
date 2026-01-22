"""
NeuroFlow Dashboard – STREAMLIT SAFE VERSION
===========================================
Fully fixed & launch-ready
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.cluster import KMeans
from datetime import datetime
import warnings

# -----------------------------------------------------------------------------
# PAGE CONFIG (MUST BE FIRST)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NeuroFlow | AI Simulation Hub",
    page_icon="🎮",
    layout="wide"
)

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/AmoghLakshman/NeuroFlow/main/neuroflow_market_survey.csv"

COLORS = {
    "primary": "#0066CC",
    "secondary": "#00C853",
    "accent": "#FF6B35"
}

# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    df["Will_Subscribe"] = df["Will_Subscribe"].str.strip().str.capitalize()
    return df

df = load_data()

# -----------------------------------------------------------------------------
# TRAIN MODELS
# -----------------------------------------------------------------------------
@st.cache_resource
def train_models(df):
    X = df[
        [
            "Age",
            "Occupation",
            "Primary_Challenge",
            "Primary_Challenge_Severity",
            "Tech_Comfort_Level",
            "Willing_To_Pay",
        ]
    ]
    y = df["Will_Subscribe"].map({"Yes": 1, "No": 0})

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), ["Age", "Primary_Challenge_Severity", "Tech_Comfort_Level", "Willing_To_Pay"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Occupation", "Primary_Challenge"]),
        ]
    )

    clf = Pipeline(
        [
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    clf.fit(X, y)

    scaler = StandardScaler()
    Xc = scaler.fit_transform(
        df[["Age", "Primary_Challenge_Severity", "Tech_Comfort_Level", "Willing_To_Pay"]]
    )

    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
    kmeans.fit(Xc)

    reg = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), ["Age", "Primary_Challenge_Severity", "Tech_Comfort_Level"]),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Occupation"]),
                    ]
                ),
            ),
            ("model", Lasso(alpha=0.1)),
        ]
    )
    reg.fit(
        df[["Age", "Primary_Challenge_Severity", "Tech_Comfort_Level", "Occupation"]],
        df["Willing_To_Pay"],
    )

    return clf, kmeans, scaler, reg


clf, kmeans, scaler, reg = train_models(df)

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Executive Summary",
        "📊 Market Intelligence",
        "🎮 AI Simulation Hub",
        "📈 Batch Predictions",
    ],
)

# -----------------------------------------------------------------------------
# PAGE: EXEC SUMMARY
# -----------------------------------------------------------------------------
if page == "🏠 Executive Summary":
    st.title("🚀 NeuroFlow Executive Dashboard")
    st.metric("Dataset Size", len(df))
    st.dataframe(df.head(), use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE: MARKET INTELLIGENCE
# -----------------------------------------------------------------------------
elif page == "📊 Market Intelligence":
    st.title("📊 Market Intelligence")

    fig = px.histogram(
        df,
        x="Age",
        color="Will_Subscribe",
        color_discrete_map={"Yes": COLORS["primary"], "No": "#CCCCCC"},
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE: AI SIMULATION HUB
# -----------------------------------------------------------------------------
elif page == "🎮 AI Simulation Hub":
    st.title("🎮 AI Simulation Hub")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 70, 35)
        occ = st.selectbox("Occupation", sorted(df["Occupation"].unique()))
        chal = st.selectbox("Challenge", sorted(df["Primary_Challenge"].unique()))
        sev = st.slider("Severity", 1, 10, 7)
        tech = st.slider("Tech Comfort", 1, 5, 4)
        wtp = st.slider("WTP", 5, 50, 25)

        predict = st.button("🚀 Predict")

    with col2:
        if predict:
            input_df = pd.DataFrame(
                {
                    "Age": [age],
                    "Occupation": [occ],
                    "Primary_Challenge": [chal],
                    "Primary_Challenge_Severity": [sev],
                    "Tech_Comfort_Level": [tech],
                    "Willing_To_Pay": [wtp],
                }
            )

            prob = clf.predict_proba(input_df)[0][1] * 100

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob,
                    title={"text": "Subscription Probability"},
                    gauge={"axis": {"range": [0, 100]}},
                )
            )
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE: BATCH PREDICTIONS
# -----------------------------------------------------------------------------
elif page == "📈 Batch Predictions":
    st.title("📈 Batch Predictions")

    uploaded = st.file_uploader("Upload CSV", type="csv")

    if uploaded:
        batch = pd.read_csv(uploaded)
        probs = clf.predict_proba(
            batch[
                [
                    "Age",
                    "Occupation",
                    "Primary_Challenge",
                    "Primary_Challenge_Severity",
                    "Tech_Comfort_Level",
                    "Willing_To_Pay",
                ]
            ]
        )[:, 1]

        batch["Probability"] = probs
        st.dataframe(batch)

        st.download_button(
            "Download Results",
            batch.to_csv(index=False),
            f"predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
        )

# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("🎮 NeuroFlow | Built with Streamlit")
