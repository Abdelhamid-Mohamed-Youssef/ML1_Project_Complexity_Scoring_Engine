
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from io import BytesIO
from fpdf import FPDF
import tempfile
import os
from weasyprint import HTML
import base64


# === Load Trained Model ===
model_path = "/Users/abdelhamidmohamed/Desktop/AI_Resource_Planning_Suite/ML-1_Resource_Planning_Scoring/output/ml1_model.pkl"
model = joblib.load(model_path)

# === Page Setup ===
st.set_page_config(page_title="Project Complexity Scoring App", layout="centered")
st.title("🏗️ Project Complexity Scoring App")
st.markdown("This app predicts project complexity using a trained ML model and compares it with your manual scoring system.")

# === Manual Input Form ===
with st.form("score_form"):
    st.subheader("🔧 Manual Project Input")

    gfa = st.number_input("GFA (sqft)", value=10000)
    floors = st.number_input("Number of Floors", value=5)
    features = st.number_input("Design Features Count", value=10)
    contractor = st.number_input("Contractor Score", value=70)
    duration = st.number_input("Construction Duration (Months)", value=12)
    elevation = st.selectbox("Elevation Complexity", ["Low", "Medium", "High"])
    logistic = st.selectbox("Logistic Limitation", ["Low", "Medium", "High"])
    voids = st.selectbox("Has Voids", ["No", "Yes"])

    submit = st.form_submit_button("Predict Score")

# === Manual Prediction ===
if submit:
    elevation_map = {"Low": 1, "Medium": 2, "High": 3}
    logistic_map = {"Low": 1, "Medium": 2, "High": 3}
    void_flag = 1 if voids == "Yes" else 0

    input_data = pd.DataFrame([{
        "GFA_sqft": gfa,
        "No_of_Floors": floors,
        "Design Features Count": features,
        "Contractor Score": contractor,
        "Construction Duration (Months)": duration,
        "Elevation_Score": elevation_map[elevation],
        "Logistic_Score": logistic_map[logistic],
        "Voids_Flag": void_flag
    }])

    prediction = model.predict(input_data)[0]
    st.success(f"✅ Predicted Project Score: {prediction:.4f}")

    if prediction <= 0.21:
        st.info("🟢 Complexity Level: Low")
    elif prediction <= 0.27:
        st.warning("🟠 Complexity Level: Medium")
    else:
        st.error("🔴 Complexity Level: High")

# === A/B Testing Section ===
st.markdown("---")
st.markdown("## 🔬 A/B Testing: Manual Scoring vs ML Prediction")
st.markdown("""
This section allows you to upload a scored Excel file and compare:
- ✅ Manual Business Logic Scores (`Total_Weighted_Score`)
- 🤖 ML Predicted Scores
- 📉 Error (MAE, R²)
- 🧮 Complexity Classifications
""")

uploaded_file = st.file_uploader("📁 Upload your project file with Total_Weighted_Score", type=["xlsx"])

if uploaded_file:
    try:
        df_ab = pd.read_excel(uploaded_file)

        required_columns = ["Project Name", "Total_Weighted_Score", "Has Voids", "Elevation Complexity", "Logistic Limitation"]
        for col in required_columns:
            if col not in df_ab.columns:
                st.error(f"❌ Missing column: `{col}` in uploaded file.")
                st.stop()

        df_ab["Voids_Flag"] = df_ab["Has Voids"].map({"Yes": 1, "No": 0})
        df_ab["Elevation_Score"] = df_ab["Elevation Complexity"].map({"Low": 1, "Medium": 2, "High": 3})
        df_ab["Logistic_Score"] = df_ab["Logistic Limitation"].map({"Low": 1, "Medium": 2, "High": 3})

        features = [
            "GFA_sqft", "No_of_Floors", "Design Features Count",
            "Contractor Score", "Construction Duration (Months)",
            "Elevation_Score", "Logistic_Score", "Voids_Flag"
        ]

        for col in features:
            if col not in df_ab.columns:
                st.error(f"❌ Missing feature column: `{col}`")
                st.stop()

        df_ab["Predicted Score"] = model.predict(df_ab[features])

        def classify(score):
            if score <= 0.21:
                return "Low"
            elif score <= 0.27:
                return "Medium"
            else:
                return "High"

        df_ab["Manual_Level"] = df_ab["Total_Weighted_Score"].apply(classify)
        df_ab["Predicted_Level"] = df_ab["Predicted Score"].apply(classify)
        df_ab["Absolute Error"] = (df_ab["Total_Weighted_Score"] - df_ab["Predicted Score"]).abs()

        # === METRICS ===
        st.subheader("📈 Evaluation Metrics")
        mae = mean_absolute_error(df_ab["Total_Weighted_Score"], df_ab["Predicted Score"])
        r2 = r2_score(df_ab["Total_Weighted_Score"], df_ab["Predicted Score"])
        st.metric("📉 MAE (Avg Error)", f"{mae:.4f}")
        st.metric("📈 R² Score", f"{r2:.2%}")

        # === VALIDATION TABLE ===
        st.subheader("🧾 Validation Matrix (First 10 Projects)")
        st.dataframe(df_ab[[
            "Project Name", "Total_Weighted_Score", "Predicted Score",
            "Absolute Error", "Manual_Level", "Predicted_Level"
        ]].head(10))

        # === SCATTERPLOT ===
        st.subheader("🔍 Manual vs Predicted Score (Scatterplot)")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df_ab, x="Total_Weighted_Score", y="Predicted Score", ax=ax1)
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel("Manual Score")
        plt.ylabel("Predicted Score")
        plt.title("Score Accuracy")
        st.pyplot(fig1)

        # === HISTOGRAM ===
        st.subheader("📉 Predicted Score Distribution")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        sns.histplot(df_ab["Predicted Score"], bins=30, kde=True, color="skyblue", ax=ax2)
        plt.title("Predicted Score Distribution")
        st.pyplot(fig2)

        # === CONFUSION MATRIX ===
        st.subheader("🧮 Confusion Matrix: Manual vs Predicted Complexity")
        st.caption("📘 Rows = Manual Labels, Columns = Predicted Labels")
        confusion = pd.crosstab(df_ab["Manual_Level"], df_ab["Predicted_Level"])
        styled_confusion = confusion.style.set_properties(**{
            'text-align': 'center',
            'font-weight': 'bold',
            'background-color': '#f9f9f9'
        }).set_table_styles([{
            'selector': 'th',
            'props': [('text-align', 'center')]
        }])
        st.dataframe(styled_confusion)

        # === DOWNLOAD SECTION ===
        st.subheader("📤 Download Results")
        validation_matrix = df_ab[[
            "Project Name", "Total_Weighted_Score", "Predicted Score",
            "Absolute Error", "Manual_Level", "Predicted_Level"
        ]]
        full_output = df_ab.copy()

        towrite = BytesIO()
        with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
            validation_matrix.to_excel(writer, sheet_name='Validation_Matrix', index=False)
            full_output.to_excel(writer, sheet_name='Full_Predictions', index=False)
        towrite.seek(0)

        st.download_button(
            label="📥 Download as Excel",
            data=towrite,
            file_name="ML1_Prediction_Results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # === PDF EXPORT ===
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, "ML1 A/B Test Summary", ln=True, align="C")

        pdf.set_font("Arial", size=12)
        pdf.ln(10)
        pdf.cell(200, 10, f"MAE (Avg Error): {mae:.4f}", ln=True)
        pdf.cell(200, 10, f"R² Score: {r2:.2%}", ln=True)
        pdf.ln(8)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, "Confusion Matrix", ln=True)

        pdf.set_font("Arial", size=11)
        for line in confusion.to_string().split("\n"):
            pdf.cell(200, 6, line, ln=True)

        tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(tmp_pdf.name)

        with open(tmp_pdf.name, "rb") as f:
            st.download_button(
                label="📄 Download PDF Summary",
                data=f,
                file_name="ML1_A_B_Report.pdf",
                mime="application/pdf"
            )

        os.unlink(tmp_pdf.name)

    except Exception as e:
        st.error(f"❌ Error processing uploaded file: {e}")
else:
    st.info("📥 Please upload a valid Excel file to begin A/B testing.")
