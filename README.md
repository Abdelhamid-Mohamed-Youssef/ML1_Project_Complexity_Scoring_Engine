# 🧠 ML1: Project Complexity Scoring Engine

This AI-powered engine was developed as an internal initiative at **DAMAC** to help management assess and prioritize projects based on multiple complexity factors. It is currently in the **UAT (User Acceptance Testing)** phase and runs in parallel with our manual scoring model. The tool enables automatic scoring, classification, and visualization of project complexity levels — with a production-ready Streamlit interface.

---

## 🎯 Purpose

> "To automate and standardize the way we score projects at DAMAC, giving Planning and Management teams a smarter way to track resource needs."

✅ Predict project complexity based on real construction data  
✅ Support data-driven decision-making in resource allocation  
✅ Provide transparency through reproducible scoring models

---

## 🛠️ Technical Stack

- **Language:** Python 3.12  
- **Libraries:** pandas, scikit-learn, matplotlib, seaborn, Streamlit  
- **App:** Streamlit dashboard (UAT-ready)  
- **Model:** Random Forest Regressor  
- **Output:** Predicted Score + Complexity Label (Low / Medium / High)

---

## 📁 Project Structure

```
ML1_Project_Complexity_Scoring_Engine/
│
├── app/                    ← Streamlit app logic
│   └── streamlit_app.py    ← Full A/B test and scoring interface
│
├── notebooks/              ← Development & experimentation
│   └── ML1_Scoring_Engine.ipynb
│
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
└── .gitignore
```

---

## 🧪 Status: UAT in Progress

This project is currently being tested internally by DAMAC's Planning & BI teams.  
✅ Models have been trained and validated  
✅ All business logic has been reviewed  
✅ Outputs were reviewed by Central Planning (✅ from Ali)

We are using a **parallel scoring model** to cross-check AI predictions vs human judgment — results will inform future scaling into ML-2 (Staff Prediction Engine).

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Streamlit app
streamlit run app/streamlit_app.py
```

Once launched, the app is available at:  
📍 http://localhost:8501

---

## 📊 Model Output

Each project is scored and labeled based on:
- GFA
- No. of Floors
- Façade complexity
- Elevation
- Voids
- Design & Logistics Scores
- Construction Value Ranking

🧮 The final score = Weighted score of all factors  
🎯 Output includes:
- Predicted Score (0–100)
- Complexity Class: Low, Medium, or High

---

## 🧩 What’s Next?

We are integrating this into:
- **ML-2: Staff Requirement & Deployment Predictor**
- **ML-3: Full Resource Planning AI Engine** combining project scoring + staff mapping

---

📬 For internal use only. For questions or improvements, contact:
**Abdelhamid Mohamed – BI Planning Team, DAMAC**



