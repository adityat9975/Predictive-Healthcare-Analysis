import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Healthcare Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
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

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
}

.sidebar-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}

/* Sidebar buttons */
.sidebar-btn button {
    width: 100%;
    text-align: left;
    background: transparent;
    color: #111827;
    padding: 0.65rem 1rem;
    border-radius: 10px;
    font-size: 0.95rem;
    border: none;
}

.sidebar-btn button:hover {
    background: #eef2ff;
    color: #4338ca;
}

/* HERO */
.hero {
    background: linear-gradient(135deg, #6366f1, #7c3aed);
    padding: 2.5rem;
    border-radius: 18px;
    color: white;
    margin-bottom: 2rem;
    text-align: center;
}

/* METRIC CARDS */
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

/* MOBILE RESPONSIVE */
@media (max-width: 768px) {
    .hero h1 {
        font-size: 1.6rem;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("👋 **Welcome, demo_user!**")
    st.button("Logout")

    st.markdown("---")
    st.markdown("🏥 **Healthcare Analytics Suite**")

    menu = [
        "📊 Dashboard",
        "🩺 Diabetes Analysis",
        "❤️ Heart Disease Analysis",
        "🧠 Parkinsons Analysis",
        "📈 Comparison Tools",
        "📄 Reports & Export",
        "💡 Health Recommendations",
        "📬 Contact & Feedback"
    ]

    selected = None
    for item in menu:
        if st.button(item, key=item):
            selected = item

# ---------------- MAIN CONTENT ----------------
st.markdown("""
<div class="hero">
    <h1>Advanced Predictive Healthcare Analytics</h1>
    <p>AI-powered disease prediction and risk assessment platform</p>
</div>
""", unsafe_allow_html=True)

st.markdown("## 📊 Healthcare Analytics Dashboard")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card"><p>Total Tests</p><h2>0</h2></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><p>Diseases Tested</p><h2>0</h2></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><p>Last Test</p><h2>Never</h2></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><p>High Risk Results</p><h2>0</h2></div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
📝 No test history available. Take your first health assessment!
</div>
""", unsafe_allow_html=True)

st.markdown("## 🚀 Quick Start")
st.write("Choose a test from the sidebar to begin your health assessment.")

