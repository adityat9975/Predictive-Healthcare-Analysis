import os
import pickle
import time
import hashlib
import re
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="AI Healthcare Risk Prediction",
    layout="wide",
    page_icon="🏥"
)

# ===============================
# MODERN UI CSS
# ===============================
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #f8fafc;
        padding: 2rem 3rem;
    }

    .hero {
        background: linear-gradient(135deg, #4f46e5, #6366f1);
        padding: 3rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
    }

    .hero h1 {
        font-size: 2.6rem;
        font-weight: 700;
    }

    .hero p {
        font-size: 1.1rem;
        opacity: 0.9;
    }

    .card {
        background: white;
        padding: 1.8rem;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.06);
        margin-bottom: 1.5rem;
    }

    .pill {
        padding: 6px 14px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    .low { background: #dcfce7; color: #166534; }
    .moderate { background: #fef3c7; color: #92400e; }
    .high { background: #fee2e2; color: #991b1b; }

    .stButton button {
        background: #4f46e5;
        color: white;
        border-radius: 12px;
        padding: 0.6rem 1.8rem;
        font-weight: 600;
        border: none;
    }

    .stButton button:hover {
        background: #4338ca;
        transform: translateY(-1px);
    }
    </style>
    """, unsafe_allow_html=True)

# ===============================
# DATA STRUCTURE
# ===============================
@dataclass
class PredictionResult:
    disease: str
    prediction: int
    probability: float
    risk_level: str
    confidence: float
    timestamp: datetime
    user_input: List[float]

# ===============================
# LOAD MODELS
# ===============================
@st.cache_resource
def load_models():
    models = {}
    if os.path.exists("model/diabetes_model.sav"):
        models["diabetes"] = pickle.load(open("model/diabetes_model.sav", "rb"))
    if os.path.exists("model/Heart_model.sav"):
        models["heart"] = pickle.load(open("model/Heart_model.sav", "rb"))
    if os.path.exists("model/parkinsons_model.sav"):
        models["parkinsons"] = pickle.load(open("model/parkinsons_model.sav", "rb"))
    return models

models = load_models()

# ===============================
# PREDICTION LOGIC
# ===============================
def predict(model_name: str, inputs: List[float]) -> PredictionResult:
    model = models[model_name]
    pred = model.predict([inputs])[0]

    if hasattr(model, "predict_proba"):
        prob = max(model.predict_proba([inputs])[0])
    else:
        prob = 0.8 if pred == 1 else 0.2

    risk = "High Risk" if pred == 1 else "Low Risk"
    return PredictionResult(
        disease=model_name,
        prediction=int(pred),
        probability=prob,
        risk_level=risk,
        confidence=prob,
        timestamp=datetime.now(),
        user_input=inputs
    )

# ===============================
# HERO SECTION
# ===============================
load_custom_css()
st.markdown("""
<div class="hero">
    <h1>AI Healthcare Risk Prediction</h1>
    <p>Early disease detection using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ===============================
# SIDEBAR NAV
# ===============================
with st.sidebar:
    st.markdown("## 🏥 Healthcare AI")
    selected = option_menu(
        None,
        ["Dashboard", "Diabetes", "Heart Disease", "Parkinson’s"],
        icons=["grid", "activity", "heart", "brain"],
        default_index=0
    )

# ===============================
# DASHBOARD
# ===============================
if selected == "Dashboard":
    st.markdown("## Dashboard Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='card'><h3>AI Models</h3><h1>3</h1></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='card'><h3>Prediction Type</h3><h1>Medical</h1></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='card'><h3>Status</h3><h1>Live</h1></div>", unsafe_allow_html=True)

# ===============================
# DIABETES
# ===============================
elif selected == "Diabetes":
    st.markdown("## 🩺 Diabetes Prediction")

    c1, c2, c3 = st.columns(3)
    with c1:
        preg = st.number_input("Pregnancies", 0, 20)
        glucose = st.number_input("Glucose", 0, 200)
        bp = st.number_input("Blood Pressure", 0, 150)
    with c2:
        skin = st.number_input("Skin Thickness", 0, 100)
        insulin = st.number_input("Insulin", 0, 900)
        bmi = st.number_input("BMI", 0.0, 70.0)
    with c3:
        dpf = st.number_input("Diabetes Pedigree", 0.0, 3.0)
        age = st.number_input("Age", 1, 120)

    if st.button("Predict Diabetes"):
        with st.spinner("Analyzing..."):
            time.sleep(1.5)
            result = predict("diabetes", [preg, glucose, bp, skin, insulin, bmi, dpf, age])

        risk_class = "high" if result.prediction else "low"

        st.markdown(f"""
        <div class="card">
            <h3>Prediction Result</h3>
            <p>Status: <b>{"Positive" if result.prediction else "Negative"}</b></p>
            <p>Confidence: <b>{result.confidence*100:.2f}%</b></p>
            <span class="pill {risk_class}">{result.risk_level}</span>
        </div>
        """, unsafe_allow_html=True)

# ===============================
# HEART
# ===============================
elif selected == "Heart Disease":
    st.markdown("## ❤️ Heart Disease Prediction")

    age = st.number_input("Age", 1, 120)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    chol = st.number_input("Cholesterol", 100, 600)

    if st.button("Predict Heart Disease"):
        with st.spinner("Analyzing..."):
            time.sleep(1.5)
            result = predict("heart", [age, sex, chol, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        risk_class = "high" if result.prediction else "low"

        st.markdown(f"""
        <div class="card">
            <h3>Prediction Result</h3>
            <p>Status: <b>{"Positive" if result.prediction else "Negative"}</b></p>
            <p>Confidence: <b>{result.confidence*100:.2f}%</b></p>
            <span class="pill {risk_class}">{result.risk_level}</span>
        </div>
        """, unsafe_allow_html=True)

# ===============================
# PARKINSONS
# ===============================
elif selected == "Parkinson’s":
    st.markdown("## 🧠 Parkinson’s Prediction")

    fo = st.number_input("MDVP:Fo(Hz)", 80.0, 300.0)
    jitter = st.number_input("Jitter", 0.0, 0.01)
    shimmer = st.number_input("Shimmer", 0.0, 0.1)

    if st.button("Predict Parkinson’s"):
        with st.spinner("Analyzing..."):
            time.sleep(1.5)
            result = predict("parkinsons", [fo, 0, 0, jitter, 0, 0, shimmer, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        risk_class = "high" if result.prediction else "low"

        st.markdown(f"""
        <div class="card">
            <h3>Prediction Result</h3>
            <p>Status: <b>{"Positive" if result.prediction else "Negative"}</b></p>
            <p>Confidence: <b>{result.confidence*100:.2f}%</b></p>
            <span class="pill {risk_class}">{result.risk_level}</span>
        </div>
        """, unsafe_allow_html=True)
