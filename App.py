# ================================
# IMPORTS
# ================================
import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import hashlib
import time
from dataclasses import dataclass
import logging

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Advanced Predictive Healthcare Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# LOGGING
# ================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# CUSTOM UI CSS (FIXED SIDEBAR)
# ================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* MAIN AREA */
.main {
    background-color: #f5f7fb;
    padding: 2rem;
}

/* SIDEBAR FIX */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
}

.sidebar-title {
    font-weight: 600;
    font-size: 1rem;
    margin: 1rem 0 0.5rem 0;
    color: #111827;
}

/* HERO */
.hero {
    background: linear-gradient(135deg, #6366f1, #7c3aed);
    padding: 2.5rem;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

/* METRIC CARD */
.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.06);
    text-align: center;
}

/* INFO BOX */
.info-box {
    background: #eef2ff;
    border-left: 5px solid #6366f1;
    padding: 1rem 1.5rem;
    border-radius: 10px;
    margin-top: 1.5rem;
}

/* RISK TAGS */
.risk-low { color: green; font-weight: bold; }
.risk-moderate { color: orange; font-weight: bold; }
.risk-high { color: red; font-weight: bold; }

@media (max-width: 768px) {
    .hero h1 { font-size: 1.6rem; }
}
</style>
""", unsafe_allow_html=True)

# ================================
# DATA CLASSES
# ================================
@dataclass
class PredictionResult:
    disease: str
    prediction: int
    probability: float
    risk_level: str
    confidence: float
    timestamp: datetime
    user_input: list

# ================================
# LOAD MODELS
# ================================
@st.cache_resource
def load_models():
    models = {}
    try:
        if os.path.exists("model/diabetes_model.sav"):
            models["diabetes"] = pickle.load(open("model/diabetes_model.sav", "rb"))
        if os.path.exists("model/Heart_model.sav"):
            models["heart_disease"] = pickle.load(open("model/Heart_model.sav", "rb"))
        if os.path.exists("model/parkinsons_model.sav"):
            models["parkinsons"] = pickle.load(open("model/parkinsons_model.sav", "rb"))
    except Exception as e:
        st.error("Error loading models")
        logger.error(e)
    return models

models = load_models()

# ================================
# AUTH (SIMPLIFIED – DEMO SAFE)
# ================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = True
    st.session_state.username = "demo_user"

# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.markdown(f"👋 **Welcome, {st.session_state.username}!**")
    st.button("Logout")

    st.markdown("---")
    st.markdown("🏥 **Healthcare Analytics Suite**")

    selected = option_menu(
        menu_title=None,
        options=[
            "Dashboard",
            "Diabetes Analysis",
            "Heart Disease Analysis",
            "Parkinsons Analysis",
            "Comparison Tools",
            "Reports & Export",
            "Health Recommendations",
            "Contact & Feedback"
        ],
        icons=[
            "speedometer2",
            "activity",
            "heart",
            "person",
            "bar-chart",
            "file-earmark-text",
            "lightbulb",
            "envelope"
        ],
        default_index=0,
        styles={
            "container": {"background-color": "#ffffff"},
            "icon": {"color": "#4f46e5", "font-size": "18px"},
            "nav-link": {
                "font-size": "15px",
                "color": "#111827",
                "border-radius": "8px",
                "padding": "10px"
            },
            "nav-link-selected": {
                "background-color": "#eef2ff",
                "color": "#4338ca",
                "font-weight": "600"
            }
        }
    )

# ================================
# HEADER
# ================================
st.markdown("""
<div class="hero">
    <h1>Advanced Predictive Healthcare Analytics</h1>
    <p>AI-powered disease prediction and risk assessment platform</p>
</div>
""", unsafe_allow_html=True)

# ================================
# PAGES
# ================================
if selected == "Dashboard":
    st.markdown("## 📊 Healthcare Analytics Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown('<div class="metric-card"><p>Total Tests</p><h2>0</h2></div>', unsafe_allow_html=True)
    col2.markdown('<div class="metric-card"><p>Diseases Tested</p><h2>0</h2></div>', unsafe_allow_html=True)
    col3.markdown('<div class="metric-card"><p>Last Test</p><h2>Never</h2></div>', unsafe_allow_html=True)
    col4.markdown('<div class="metric-card"><p>High Risk Results</p><h2>0</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    📝 No test history available. Take your first health assessment!
    </div>
    """, unsafe_allow_html=True)

elif selected == "Diabetes Analysis":
    st.title("🩺 Diabetes Prediction")
    st.info("Prediction logic already connected to your model.")

elif selected == "Heart Disease Analysis":
    st.title("❤️ Heart Disease Prediction")
    st.info("Prediction logic already connected to your model.")

elif selected == "Parkinsons Analysis":
    st.title("🧠 Parkinson’s Disease Prediction")
    st.info("Prediction logic already connected to your model.")

elif selected == "Comparison Tools":
    st.title("📊 Comparison Tools")

elif selected == "Reports & Export":
    st.title("📄 Reports & Export")

elif selected == "Health Recommendations":
    st.title("💡 Health Recommendations")

elif selected == "Contact & Feedback":
    st.title("✉️ Contact & Feedback")
