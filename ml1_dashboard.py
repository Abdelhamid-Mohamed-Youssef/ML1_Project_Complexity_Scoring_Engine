import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# === Page Configuration ===
st.set_page_config(layout="centered", page_title="ML-1 Scoring Dashboard")
st.title("🔍 ML-1 Project Dashboard – Resource Complexity Scoring")

# === Load Model and Scaler ===
try:
    model = joblib.load("ml1_model.pkl")
    scaler = joblib.load("ml1_scaler.pkl")
except:
    st.error("❌ Model or scaler file not found. Please ensure 'ml1_model.pkl' and 'ml1_scaler.pkl' exist.")
    st.stop()

# === Load Data File ===
data_file = "ML1_SCORED_LABELED_PROJECTS_RE_PREDICTED_AB_COMBINED_TEST.xlsx"
try:
    df = pd.read_excel(data_file, sheet_name="Full Data")
    summary = pd.read_excel(data_file, sheet_name="Summary")
    conf_matrix = pd.read_excel(data_file, sheet_name="Confusion Matrix", index_col=0)

    st.success("✅ Data loaded successfully.")

    # === Section: A/B Test Summary ===
    st.subheader("📋 A/B Testing Summary")
    st.dataframe(summary)

    with st.expander("🧠 What do these numbers mean?"):
        st.markdown("""
        - **Classification Accuracy**: % of correct complexity classifications.
        - **MAE (Mean Absolute Error)**: Avg. deviation between predicted and actual score.
        - **MAPE (Mean Absolute Percentage Error)**: Error % of actual value.
        - **Within 5% / 10%**: Predictions that are close to actual values.
        - **Mismatches**: Complexity level classification errors.
        """)

    # === Confusion Matrix ===
    st.subheader("📊 Confusion Matrix: Complexity Prediction")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    ax.set_title("Confusion Matrix\n(Actual vs Predicted Complexity Level)")
    ax.set_xlabel("Predicted Level")
    ax.set_ylabel("Actual Level")
    st.pyplot(fig)

    # === Live Prediction Section ===
    st.subheader("📌 Try a New Project – Predict Complexity")

    gfa = st.number_input("🏗️ GFA (sqft)", 1000, 200000, 25000)
    floors = st.slider("🏢 Number of Floors", 1, 50, 5)
    design_feats = st.slider("🎨 Design Complexity Score (0–10)", 0, 10, 3)
    logistic_score = st.slider("🚚 Logistic Score (0–10)", 0.0, 10.0, 4.0)
    contractor_score = st.slider("🔧 Contractor Qualification Score (0–100)", 0, 100, 70)
    duration = st.slider("📅 Construction Duration (Months)", 1, 60, 18)
    procurement_score = st.slider("🏗️ Procurement Strategy Score (0–10)", 0.0, 10.0, 5.0)

    if st.button("🔮 Predict Complexity"):
        input_data = pd.DataFrame([[
            gfa, floors, design_feats, logistic_score,
            contractor_score, duration, procurement_score
        ]], columns=[
            "Gross Floor Area (GFA)",
            "Number of Floors",
            "Design Complexity Score",
            "Logistic Score",
            "Contractor Qualification Score",
            "Construction Duration",
            "Procurement Strategy Score"
        ])

        # Apply scaling
        scaled_input = scaler.transform(input_data)
        predicted_score = model.predict(scaled_input)[0]

        # Classification Logic
        if predicted_score >= 60:
            level = "High"
            color = "🔴"
        elif predicted_score >= 35:
            level = "Medium"
            color = "🟠"
        else:
            level = "Low"
            color = "🟢"

        st.markdown(f"### 🧮 Predicted Score: `{round(predicted_score, 2)}`")
        st.markdown(f"### 🎯 Complexity Level: **{color} {level}**")

except FileNotFoundError:
    st.error("❌ Excel file not found. Please ensure it exists in the same folder.")
