import streamlit as st
from components.banner import render_banner

# ---------- Page Config ----------
st.set_page_config(
    page_title="Exoplanet Hunter",
    page_icon="🌌",
    layout="wide",
)
render_banner()
hide_streamlit_header_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_header_style, unsafe_allow_html=True)

# ---------- Custom CSS ----------
st.markdown("""
<style>
/* --- App base --- */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 50% 50%, #0b0f1a 0%, #0e162a 60%, #0a0e27 100%);
    color: #e0e0e0;
    font-family: 'Inter', 'Open Sans', sans-serif;
    overflow: hidden;
}

/* --- Transit background layer --- */
.transit-bg {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    z-index: 0;
    pointer-events: none;
}

/* --- Star --- */
.star {
    position: absolute;
    top: 40%; left: 50%;
    transform: translate(-50%, -50%);
    width: 250px; height: 250px;
    border-radius: 50%;
    background: radial-gradient(circle at 30% 30%, #fff8e1, #f5b971 70%, #9e6c33 100%);
    box-shadow: 0 0 80px 30px rgba(255, 215, 150, 0.3);
}

/* --- Planet transit (12s linear) --- */
.planet {
    position: absolute;
    top: 40%; left: 100%;
    width: 60px; height: 60px;
    border-radius: 50%;
    background: radial-gradient(circle at 30% 30%, #1a1a1a, #333);
    animation: transit 12s linear infinite;
}
@keyframes transit {
    0% { left: 110%; opacity: 0; }
    10% { opacity: 1; }
    50% { left: 50%; opacity: 1; } /* 行星在中心 */
    90% { opacity: 1; }
    100% { left: -10%; opacity: 0; }
}

/* --- Global dimmer synced with transit (12s linear) --- */
.dimmer {
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0);
    pointer-events: none;
    z-index: 1;
    animation: dimmer 12s linear infinite;
}
@keyframes dimmer {
    0%   { background: rgba(0,0,0,0); }
    40%  { background: rgba(0,0,0,0); }
    45%  { background: rgba(0,0,0,0.40); }  /* 亮度降低40% */
    55%  { background: rgba(0,0,0,0.40); }
    60%  { background: rgba(0,0,0,0); }
    100% { background: rgba(0,0,0,0); }
}

/* --- Front content --- */
.front-content {
    position: relative;
    z-index: 2;
}

/* --- Title --- */
.space-title {
    font-size: 3.2rem;
    font-weight: 800;
    text-align: center;
    color: #ffffff;
    letter-spacing: .08em;
    margin-top: 2rem;
    text-shadow: 0 0 20px rgba(100,150,255,0.6);
}

/* --- Subtitle --- */
.subtitle {
    text-align: center;
    font-size: 1.4rem;
    color: #ffffff;
    margin-bottom: 2.5rem;
}

/* --- Metric card text (改成白色) --- */
[data-testid="stMetric"] label,
[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

/* --- Mission text --- */
.metric-card p {
    font-size: 1.25rem;
    color: #ffffff;
    line-height: 1.7;
}

/* --- Metric cards --- */
.metric-card {
    background: rgba(20, 30, 60, 0.7);
    border: 1px solid rgba(80, 120, 200, 0.4);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

/* --- Buttons --- */
.stButton>button {
    background: linear-gradient(90deg, #3fa9f5, #5271ff);
    color: white;
    border-radius: 10px;
    font-weight: 800;
    font-size: 1.3rem;
    transition: all .3s ease;
    padding: 0.6rem 1rem;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 25px rgba(80,120,255,0.6);
}

/* --- Footer --- */
.footer {
    text-align: center;
    color: #aaa;
    padding: 1rem;
    font-size: 0.95rem;
}

/* --- Responsive title --- */
@media (max-width: 768px) {
    .space-title { font-size: 2rem; }
    .subtitle { font-size: 1.1rem; }
}
</style>
""", unsafe_allow_html=True)

# ---------- Background HTML ----------
st.markdown("""
<div class="transit-bg">
    <div class="star"></div>
    <div class="planet"></div>
    <div class="dimmer"></div>
</div>
""", unsafe_allow_html=True)

# ---------- Front content ----------
st.markdown('<div class="front-content">', unsafe_allow_html=True)
st.markdown('<h1 class="space-title">EXOPLANET HUNTER</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI × Human Collaboration Platform for Professional Exoplanet Vetting</p>', unsafe_allow_html=True)

# --- Metrics ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Candidates Loaded", "12,430")
c2.metric("AI-Flagged", "2,310")
c3.metric("Human-Vetted", "7,945")
c4.metric("Confirmed", "124")

# --- Mission card ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="metric-card">
    <h3 style="color:#3fa9f5;">Our Mission</h3>
    <p>
        Combining <b>AI precision</b> with <b>human intuition</b> to accelerate the discovery of new exoplanets.
        Collaborate, verify, and contribute to a growing database of confirmed transits.
    </p>
</div>
""", unsafe_allow_html=True)

# --- Buttons ---
b1, b2, b3 = st.columns(3)
with b1:
    st.button("Start Vetting", use_container_width=True, key="btn_start_vetting")
with b2:
    st.button("Candidate Database", use_container_width=True, key="btn_db")
with b3:
    st.button("User Contributions", use_container_width=True, key="btn_user")

# --- Footer ---
st.markdown('<br><br>', unsafe_allow_html=True)
st.markdown('<div class="footer">Exoplanet Hunter v3.1 — Dynamic Transit Background</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
