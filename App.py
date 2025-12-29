import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import hashlib
from typing import Dict, List, Tuple
import logging
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB connection (optional)
try:
    from pymongo import MongoClient
    mongo_uri = os.getenv("MONGO_URI")
    if mongo_uri:
        client = MongoClient(mongo_uri)
        db = client['HealthcareDB']
        feedback_collection = db['Feedbacks']
        medical_input_collection = db['MedicalInputs']
        users_collection = db['Users']
    else:
        client = None
        feedback_collection = medical_input_collection = users_collection = None
except:
    client = None
    feedback_collection = medical_input_collection = users_collection = None

@dataclass
class PredictionResult:
    disease: str
    prediction: int
    probability: float
    risk_level: str
    confidence: float
    timestamp: datetime
    user_input: List[float]

# ============ MODERN CSS STYLING ============
def load_modern_css():
    """Load modern, gradient-based UI styling"""
    st.markdown("""
    <style>
    /* Root variables */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --success-color: #4CAF50;
        --warning-color: #FF9800;
        --danger-color: #f44336;
        --light-bg: #f8f9fa;
        --dark-text: #1a1a1a;
    }

    /* Main container */
    .main {
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }

    /* Header styling with glassmorphism */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        backdrop-filter: blur(10px);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
        animation: slideDown 0.6s ease-out;
    }

    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }

    .main-header p {
        font-size: 1.1rem;
        opacity: 0.95;
        margin: 0;
    }

    /* Card styling with hover effects */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border-left: 5px solid var(--primary-color);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin: 0.5rem 0;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 25px rgba(0, 0, 0, 0.15);
    }

    .prediction-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.2);
        margin: 1rem 0;
        backdrop-filter: blur(5px);
    }

    /* Risk level badges with animations */
    .risk-badge {
        display: inline-block;
        padding: 0.6rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
        animation: pulse 2s infinite;
    }

    .risk-low {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
    }

    .risk-moderate {
        background: linear-gradient(135deg, #FF9800, #f57c00);
        color: white;
    }

    .risk-high {
        background: linear-gradient(135deg, #f44336, #d32f2f);
        color: white;
    }

    /* Input styling */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextInput > div > div > input {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 0.75rem;
        transition: all 0.3s ease;
        font-size: 1rem;
    }

    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }

    .stButton > button:active {
        transform: translateY(-1px);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    /* Metric display */
    [data-testid="metric-container"] {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border-left: 4px solid var(--primary-color);
    }

    /* Info/success/error messages */
    .stAlert {
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid;
        backdrop-filter: blur(5px);
    }

    /* Animations */
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.85; }
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    /* Section titles */
    h2, h3 {
        color: var(--dark-text);
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    /* Markdown text */
    .markdown-text {
        font-size: 1.05rem;
        line-height: 1.6;
        color: #333;
    }

    /* Progress indicators */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }

    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, #667eea 0%, transparent 50%, #764ba2 100%);
        margin: 2rem 0;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            padding: 2rem 1rem;
        }

        .main-header h1 {
            font-size: 1.8rem;
        }

        .metric-card {
            margin: 0.5rem 0;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Healthcare Analytics",
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="expanded"
)

# ============ MODEL LOADING ============
@st.cache_resource
def load_models():
    models = {}
    MODEL_DIR = "model"
    try:
        model_files = {
            'diabetes': 'diabetes_model.sav',
            'heart_disease': 'Heart_model.sav',
            'parkinsons': 'parkinsons_model.sav'
        }
        for key, filename in model_files.items():
            path = os.path.join(MODEL_DIR, filename)
            if os.path.exists(path):
                models[key] = pickle.load(open(path, 'rb'))
        return models
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return {}

# ============ PREDICTION MANAGER ============
class PredictionManager:
    def __init__(self, models: Dict):
        self.models = models

    def predict(self, model_name, user_input):
        model = self.models.get(model_name)
        if not model:
            return None

        try:
            prediction = model.predict([user_input])[0]
            prob = getattr(model, 'predict_proba', lambda x: [[0.6, 0.4]])([user_input])[0]
            confidence = max(prob)

            risk_map = {
                'diabetes': 'High Risk' if prediction else 'Low Risk',
                'heart_disease': 'High Risk' if prediction else 'Low Risk',
                'parkinsons': 'High Risk' if prediction else 'Low Risk'
            }

            return PredictionResult(
                disease=model_name.replace('_', ' ').title(),
                prediction=int(prediction),
                probability=max(prob),
                risk_level=risk_map.get(model_name, 'Unknown'),
                confidence=confidence,
                timestamp=datetime.now(),
                user_input=user_input
            )
        except:
            return None

# ============ VISUALIZATION MANAGER ============
class VisualizationManager:
    @staticmethod
    def create_gauge(score: float, level: str):
        colors = {'Low Risk': '#4CAF50', 'Moderate Risk': '#FF9800', 'High Risk': '#f44336'}
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            title={"text": f"Risk Level: {level}"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': colors.get(level, '#FF9800')},
                'steps': [
                    {'range': [0, 33], 'color': "#e8f5e9"},
                    {'range': [33, 66], 'color': "#fff3e0"},
                    {'range': [66, 100], 'color': "#ffebee"}
                ]
            }
        ))
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=70, b=20))
        return fig

    @staticmethod
    def create_bar_chart(values: List[float], names: List[str], title: str):
        df = pd.DataFrame({'Feature': names, 'Value': values})
        fig = px.bar(df, x='Feature', y='Value', title=title, 
                     color='Value', color_continuous_scale='Viridis')
        fig.update_layout(height=400, xaxis_tickangle=-45)
        return fig

# ============ AUTH MANAGER ============
class AuthManager:
    @staticmethod
    def hash_password(password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def authenticate(username: str, password: str) -> bool:
        if not users_collection:
            return True
        user = users_collection.find_one({"username": username})
        return user and AuthManager.hash_password(password) == user.get('password_hash', '')

# ============ MAIN RENDERING FUNCTIONS ============
def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Healthcare Analytics</h1>
        <p>AI-Powered Disease Prediction & Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)

def render_dashboard():
    st.markdown("## 📊 Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tests", 12, delta="+2")
    with col2:
        st.metric("Diseases Tested", 3)
    with col3:
        st.metric("Last Test", "Today")
    with col4:
        st.metric("High Risk", 2)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🚀 Quick Start")
        if st.button("🩺 Diabetes Check", use_container_width=True):
            st.session_state.page = "Diabetes"
            st.rerun()
        if st.button("❤️ Heart Check", use_container_width=True):
            st.session_state.page = "Heart"
            st.rerun()
        if st.button("🧠 Parkinson Check", use_container_width=True):
            st.session_state.page = "Parkinsons"
            st.rerun()

    with col2:
        st.markdown("### 📈 Quick Stats")
        stats = pd.DataFrame({
            'Status': ['Positive', 'Negative'],
            'Count': [3, 9]
        })
        fig = px.pie(stats, values='Count', names='Status', title="Results Distribution")
        st.plotly_chart(fig, use_container_width=True)

def render_diabetes():
    st.markdown("## 🩺 Diabetes Prediction")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = st.number_input('Pregnancies', 0, 17, 0)
        glucose = st.number_input('Glucose', 0, 200, 120)
        bp = st.number_input('Blood Pressure', 0, 122, 70)
    with col2:
        skin = st.number_input('Skin Thickness', 0, 99, 20)
        insulin = st.number_input('Insulin', 0, 846, 79)
        bmi = st.number_input('BMI', 0.0, 67.1, 32.0)
    with col3:
        pedigree = st.number_input('Pedigree', 0.0, 2.42, 0.47)
        age = st.number_input('Age', 0, 120, 30)

    user_input = [pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age]
    
    if st.button("🔬 Predict", use_container_width=True, type="primary"):
        with st.spinner("Analyzing..."):
            time.sleep(1)
            result = PredictionManager(models).predict('diabetes', user_input)
            if result:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(VisualizationManager.create_gauge(result.probability, result.risk_level))
                with col2:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>Results</h3>
                        <p><b>Prediction:</b> {'Positive' if result.prediction else 'Negative'}</p>
                        <p><b>Confidence:</b> {result.confidence:.1%}</p>
                        <p><span class="risk-badge risk-{result.risk_level.split()[0].lower()}">{result.risk_level}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                st.success("✅ Analysis complete!")

def render_heart():
    st.markdown("## ❤️ Heart Disease Prediction")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input('Age', 1, 120, 52)
        sex = st.selectbox('Sex', ['Male', 'Female'])
        cp = st.selectbox('Chest Pain', list(range(4)))
        trestbps = st.number_input('Resting BP', 80, 200, 120)
    with col2:
        chol = st.number_input('Cholesterol', 100, 600, 230)
        fbs = st.selectbox('Fasting BS', ['No', 'Yes'])
        restecg = st.selectbox('Resting ECG', list(range(3)))
        thalach = st.number_input('Max HR', 70, 220, 150)
    with col3:
        exang = st.selectbox('Exercise Angina', ['No', 'Yes'])
        oldpeak = st.number_input('ST Depression', 0.0, 6.2, 1.0)
        slope = st.selectbox('Slope', list(range(3)))
        ca = st.number_input('Major Vessels', 0, 4, 0)
    with col4:
        thal = st.selectbox('Thal', list(range(4)))

    user_input = [age, 1 if sex == 'Male' else 0, cp, trestbps, chol, 
                  1 if fbs == 'Yes' else 0, restecg, thalach, 
                  1 if exang == 'Yes' else 0, oldpeak, slope, ca, thal]
    
    if st.button("🔬 Predict", use_container_width=True, type="primary"):
        with st.spinner("Analyzing..."):
            time.sleep(1)
            result = PredictionManager(models).predict('heart_disease', user_input)
            if result:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(VisualizationManager.create_gauge(result.probability, result.risk_level))
                with col2:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>Results</h3>
                        <p><b>Prediction:</b> {'Positive' if result.prediction else 'Negative'}</p>
                        <p><b>Confidence:</b> {result.confidence:.1%}</p>
                        <p><span class="risk-badge risk-{result.risk_level.split()[0].lower()}">{result.risk_level}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                st.success("✅ Analysis complete!")

def render_parkinsons():
    st.markdown("## 🧠 Parkinson's Prediction")
    st.info("Enter 20 voice feature parameters")
    
    col1, col2, col3, col4 = st.columns(4)
    inputs = []
    
    features = ['Fo', 'Fhi', 'Flo', 'Jitter%', 'JitterAbs', 'RAP', 'PPQ', 'Shimmer', 
                'ShimmerDB', 'APQ3', 'APQ5', 'APQ', 'DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'D2', 'PPE']
    
    ranges = [(88, 260), (100, 600), (65, 200), (0, 0.05), (0, 0.0005), (0, 0.004), (0, 0.004),
              (0, 0.1), (0, 1.5), (0, 0.05), (0, 0.05), (0, 0.07), (0, 0.01), (0, 0.05), (10, 35),
              (0.3, 0.7), (0.5, 0.9), (1, 4), (0, 0.6)]
    
    defaults = [150, 200, 100, 0.005, 0.0001, 0.001, 0.001, 0.04, 0.4, 0.02, 0.02, 0.03, 0.005, 0.02, 20, 0.5, 0.7, 2, 0.2]
    
    cols = [col1, col2, col3, col4]
    for i, (feat, (mn, mx), default) in enumerate(zip(features, ranges, defaults)):
        col_idx = i % 4
        if col_idx == 0:
            col1, col2, col3, col4 = st.columns(4)
        cols = [col1, col2, col3, col4]
        with cols[col_idx]:
            inputs.append(st.number_input(feat, float(mn), float(mx), float(default)))
    
    if st.button("🔬 Predict", use_container_width=True, type="primary"):
        with st.spinner("Analyzing..."):
            time.sleep(1)
            result = PredictionManager(models).predict('parkinsons', inputs)
            if result:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(VisualizationManager.create_gauge(result.probability, result.risk_level))
                with col2:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>Results</h3>
                        <p><b>Prediction:</b> {'Positive' if result.prediction else 'Negative'}</p>
                        <p><b>Confidence:</b> {result.confidence:.1%}</p>
                        <p><span class="risk-badge risk-{result.risk_level.split()[0].lower()}">{result.risk_level}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                st.success("✅ Analysis complete!")

# ============ MAIN APP ============
load_modern_css()

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Dashboard'
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = True
    st.session_state.username = "User"

# Skip auth if no DB
if not users_collection:
    st.session_state.authenticated = True

render_header()

# Load models
models = load_models()

# Sidebar navigation
with st.sidebar:
    st.markdown("### 🔍 Navigation")
    page = option_menu(
        "Menu",
        ["Dashboard", "Diabetes", "Heart", "Parkinsons", "About"],
        icons=["speedometer2", "activity", "heart", "person", "info-circle"],
        default_index=0
    )

# Route pages
if page == "Dashboard":
    render_dashboard()
elif page == "Diabetes":
    render_diabetes()
elif page == "Heart":
    render_heart()
elif page == "Parkinsons":
    render_parkinsons()
elif page == "About":
    st.markdown("## ℹ️ About")
    st.info("""
    **Healthcare Analytics Platform**
    
    This AI-powered platform helps predict health risks for:
    - 🩺 Diabetes
    - ❤️ Heart Disease
    - 🧠 Parkinson's Disease
    
    Predictions are based on machine learning models trained on real medical data.
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ for healthcare")
