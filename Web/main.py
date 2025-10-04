"""
🌌 Professional Exoplanet Vetting Platform
AI × Human Collaboration System
"""

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
    .stApp { background: linear-gradient(135deg, #0a0e27 0%, #16213e 50%, #0f3460 100%); color: #e0e0e0; }
    .stApp::before { content:""; position:fixed; top:0; left:0; width:100%; height:100%;
        background-image:
          radial-gradient(2px 2px at 20px 30px, #eee, rgba(0,0,0,0)),
          radial-gradient(2px 2px at 60px 70px, #fff, rgba(0,0,0,0)),
          radial-gradient(1px 1px at 50px 50px, #fff, rgba(0,0,0,0));
        background-repeat:repeat; background-size:200px 200px; opacity:.4; z-index:-1;
        animation: twinkle 3s ease-in-out infinite; }
    @keyframes twinkle { 0%, 100% { opacity: .3; } 50% { opacity: .6; } }
    .space-title {
        font-size: 4rem; font-weight: 900; text-align: center;
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 300% 300%; -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        animation: gradient-shift 3s ease infinite; padding: 2rem 0; letter-spacing: .1em;
    }
    @keyframes gradient-shift { 0%,100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
    .candidate-card {
        background: linear-gradient(135deg, rgba(22,33,62,.95), rgba(15,52,96,.95));
        border: 2px solid rgba(102,126,234,.3); border-radius: 20px; padding: 2rem; margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(31,38,135,.37); backdrop-filter: blur(10px);
    }
    .metric-card {
        background: rgba(22,33,62,.6); border: 1px solid rgba(102,126,234,.3); border-radius: 15px;
        padding: 1.5rem; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,.3);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 15px; padding: 1rem 2rem; font-size: 1.1rem; font-weight: 600;
        box-shadow: 0 4px 15px rgba(102,126,234,.4); transition: all .3s ease;
    }
    .stButton > button:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(102,126,234,.6); }
</style>
""", unsafe_allow_html=True)

# ---------- Helper: navigate to pages/vetting.py ----------
def go_to_vetting():
    """
    Tries (1) native st.switch_page, (2) streamlit-extras switch_page_button,
    else (3) query-params fallback (requires your pages read st.query_params).
    """
    # 1) Native Streamlit (>=1.25-ish)
    if hasattr(st, "switch_page"):
        try:
            st.switch_page("pages/vetting.py")
            return
        except Exception:
            pass

    # 2) Community extras
    try:
        from streamlit_extras.switch_page_button import switch_page
        switch_page("vetting")   # uses page title/filename without .py
        return
    except Exception:
        pass

    # 3) Fallback via query params (works if your multipage logic reads it)
    try:
        st.query_params.update(page="vetting")
    except Exception:
        st.experimental_set_query_params(page="vetting")
    st.rerun()

# ---------- MAIN PAGE ----------
st.markdown('<h1 class="space-title">🌌 EXOPLANET HUNTER</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;font-size:1.5rem;color:#a0a0a0;letter-spacing:.2em;">'
    'AI × HUMAN COLLABORATION PLATFORM</p>', unsafe_allow_html=True
)

st.markdown("""
<div class="candidate-card">
  <h2 style='text-align:center;color:#667eea;'>✨ OUR MISSION ✨</h2>
  <p style='text-align:center;font-size:1.3rem;line-height:1.8;margin-top:1rem;'>
    Combining <strong>AI's Speed</strong> with <strong>Human Intuition</strong><br>
    To discover new worlds in our universe
  </p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="metric-card">
        <div style='font-size:3rem;margin-bottom:.5rem;'>🤖</div>
        <h3 style='color:#667eea;'>AI Screening</h3>
        <p style='color:#a0a0a0;'>Rapid analysis<br>Signal detection<br>Uncertainty flagging</p>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="metric-card">
        <div style='font-size:3rem;margin-bottom:.5rem;'>👁️</div>
        <h3 style='color:#4facfe;'>Human Vetting</h3>
        <p style='color:#a0a0a0;'>Expert review<br>Intuitive swipes<br>Critical judgment</p>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="metric-card">
        <div style='font-size:3rem;margin-bottom:.5rem;'>🔄</div>
        <h3 style='color:#f093fb;'>Collaborative Learning</h3>
        <p style='color:#a0a0a0;'>Model improvement<br>Contribution tracking<br>Continuous evolution</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# 🔗 Button -> pages/vetting.py
if st.button("🚀 START VETTING", use_container_width=True, type="primary"):
    go_to_vetting()

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;color:#667eea;padding:2rem;'>
  <p style='font-size:.9rem;opacity:.6;'>🌌 EXOPLANET HUNTER v2.1 | Professional Vetting Platform</p>
</div>
""", unsafe_allow_html=True)
