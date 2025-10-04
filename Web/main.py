# ======================================================
# 🌌 EXOPLANET HUNTER v2.1 — Professional Vetting Platform
# ======================================================

import streamlit as st
import numpy as np

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="Exoplanet Hunter",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styles ----------
st.markdown("""
<style>
    /* Background + Global Colors */
    .stApp {
        background: radial-gradient(circle at 20% 30%, #0b0f1a 0%, #0e1630 50%, #0a0e27 100%);
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    /* Subtle star field overlay */
    .stApp::before {
        content:""; position:fixed; top:0; left:0; width:100%; height:100%;
        background-image:
          radial-gradient(1px 1px at 20px 30px, #3fa9f5, rgba(0,0,0,0)),
          radial-gradient(1px 1px at 60px 70px, #ffffff, rgba(0,0,0,0)),
          radial-gradient(1px 1px at 90px 120px, #a0a0a0, rgba(0,0,0,0));
        background-repeat:repeat; background-size:200px 200px;
        opacity:.25; z-index:-1;
    }
    /* Title gradient */
    .space-title {
        font-size: 3.5rem; font-weight: 800; text-align: center;
        background: linear-gradient(45deg, #3fa9f5, #4facfe, #00f2fe);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: .1em; margin-bottom: 1rem;
    }
    /* KPI metrics */
    .metric-container {
        background: rgba(20, 30, 60, 0.7);
        border: 1px solid rgba(63,169,245,.4);
        border-radius: 12px; padding: 1.5rem; text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,.4);
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #3fa9f5; }
    .metric-label { color: #a0a0a0; font-size: 1rem; }
    /* Card section */
    .info-card {
        background: rgba(15, 25, 45, 0.9);
        border: 1px solid rgba(63,169,245,.3);
        border-radius: 15px; padding: 2rem;
        text-align: center; box-shadow: 0 4px 25px rgba(0,0,0,.4);
        margin-top: 2rem;
    }
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3fa9f5, #4facfe);
        color: white; border: none; border-radius: 12px;
        padding: 1rem 2rem; font-size: 1.1rem; font-weight: 600;
        box-shadow: 0 4px 15px rgba(63,169,245,.4); transition: all .3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(63,169,245,.6);
    }
    /* Footer */
    .footer {
        text-align:center; color:#555; font-size:.9rem; margin-top:3rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------- Navigation Helper ----------
def go_to_vetting():
    """Navigate to pages/vetting.py"""
    if hasattr(st, "switch_page"):
        try:
            st.switch_page("pages/vetting.py")
            return
        except Exception:
            pass
    try:
        from streamlit_extras.switch_page_button import switch_page
        switch_page("vetting")
        return
    except Exception:
        pass
    try:
        st.query_params.update(page="vetting")
    except Exception:
        st.experimental_set_query_params(page="vetting")
    st.rerun()


# ---------- MAIN PAGE ----------
st.markdown('<h1 class="space-title">EXOPLANET HUNTER</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;font-size:1.2rem;color:#b0b0b0;">'
    'A Professional Exoplanet Vetting Platform — powered by <b>AI × Human Collaboration</b></p>',
    unsafe_allow_html=True
)

# ----- KPI Metrics -----
st.markdown("<br>", unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown('<div class="metric-container"><div class="metric-value">12,430</div><div class="metric-label">Candidates Loaded</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown('<div class="metric-container"><div class="metric-value">2,310</div><div class="metric-label">AI-Flagged</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown('<div class="metric-container"><div class="metric-value">7,945</div><div class="metric-label">Human-Vetted</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown('<div class="metric-container"><div class="metric-value">124</div><div class="metric-label">Confirmed</div></div>', unsafe_allow_html=True)

# ----- Mission Section -----
st.markdown("""
<div class="info-card">
  <h2 style='color:#3fa9f5;'>Our Mission</h2>
  <p style='font-size:1.1rem;line-height:1.7;color:#ddd;'>
  Combining the <b>Speed of AI</b> with the <b>Insight of Astronomers</b> <br>
  to accelerate the discovery of <b>new worlds</b> beyond our solar system.
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----- Navigation Buttons -----
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🚀 Start Vetting", use_container_width=True):
        go_to_vetting()
with col2:
    st.button("📊 Candidate Database", use_container_width=True)
with col3:
    st.button("👥 Contributions", use_container_width=True)

# ----- Footer -----
st.markdown("""
<div class="footer">
🌌 Exoplanet Hunter v2.1 | Inspired by ExoClock & NASA Exoplanet Archive
</div>
""", unsafe_allow_html=True)
