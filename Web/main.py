import streamlit as st
from components.banner import render_banner

# ---------- Page Config ----------
st.set_page_config(
    page_title="ExoMatch - Exoplanet Hunter",
    page_icon="Web/logo.png",
    layout="wide",
)

render_banner()
st.markdown("""
    <style>
    #MainMenu, footer, header {visibility:hidden;}
    </style>
""", unsafe_allow_html=True)

# ---------- CSS ----------
st.markdown("""
<style>
/* ===== 基礎設定 ===== */
[data-testid="stAppViewContainer"]{
  background: radial-gradient(circle at 50% 50%, #0b0f1a 0%, #0e162a 60%, #0a0e27 100%);
  color:#eaeef7; font-family:'Inter','Open Sans',sans-serif; overflow-x:hidden;
}

/* ===== 背景動畫 ===== */
.transit-bg{ position:fixed; inset:0; z-index:0; pointer-events:none; filter:brightness(1); transition:filter .15s linear; }
.star{
  position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
  width:360px; height:360px; border-radius:50%;
  background:radial-gradient(circle at 32% 30%, #fff8e1, #f5b971 70%, #9e6c33 100%);
  box-shadow:0 0 120px 48px rgba(255,215,150,.28);
}
.planet{
  position:absolute; top:50%; transform:translateY(-50%); left:105%;
  width:72px; height:72px; border-radius:50%;
  background:radial-gradient(circle at 30% 30%, #1a1a1a, #333);
  animation:transit 12s linear infinite;
}
@keyframes transit{
  0%{left:110%;opacity:0;} 8%{opacity:1;} 50%{left:50%;opacity:1;}
  92%{opacity:1;} 100%{left:-10%;opacity:0;}
}
.dimmer{ position:fixed; inset:0; z-index:1; pointer-events:none; background:rgba(0,0,0,0); animation:dimmer 12s linear infinite; }
@keyframes dimmer{ 0%,38%{background:rgba(0,0,0,0);} 45%,55%{background:rgba(0,0,0,.2);} 62%,100%{background:rgba(0,0,0,0);} }

/* ===== 全頁暗化層 ===== */
.global-dimmer{
  position:fixed; inset:0; z-index:100000; pointer-events:none;
  background:rgba(0,0,0,0); transition:background .16s ease;
}

/* ===== HERO ===== */
.hero{
  min-height:86vh;
  display:flex; flex-direction:column; align-items:center; text-align:center;
  justify-content:center;
  padding-inline:2rem; gap:0rem;
}
.space-title{
  margin:0; color:#fff; font-weight:900; letter-spacing:.1em;
  font-size:clamp(4.2rem,12vw,9rem);
  text-shadow:0 2px 6px rgba(0,0,0,.35), 0 0 3px rgba(0,0,0,.35);
}
.subtitle{
  margin:.3rem 0 0 0; color:#f5f7ff; opacity:.96; font-weight:700; letter-spacing:.08em;
  font-size:clamp(1.6rem,3.5vw,2.6rem);
  text-shadow:0 2px 6px rgba(0,0,0,.30), 0 0 3px rgba(0,0,0,.30);
}

/* ===== 專屬上方 CTA 按鈕樣式 ===== */
.cta-wrap{ margin-top: clamp(6vh, 8vh, 10vh); display:flex; justify-content:center; align-items:center; }
.cta-button{
  display:inline-flex;
  align-items:center;
  justify-content:center;
  width:270px;
  height:78px;
  background:transparent;
  color:#ffffff;
  border:2px solid rgba(235,238,245,0.9);
  border-radius:0;
  font-weight:900;
  font-size:1.3rem;
  letter-spacing:.06em;
  cursor:pointer;
  transition: transform .22s ease, background .22s ease, color .22s ease, border-color .22s ease;
}
.cta-button:hover{
  background:#ffffff;
  color:#0c1225;
  border-color:#ffffff;
  transform:translateY(-2px);
}

/* ===== 下面一般按鈕樣式（保持原樣） ===== */
.stButton > button {
  background: rgba(20, 30, 60, 0.85) !important;
  color: #ffffff !important;
  border: 2px solid rgba(80, 120, 200, 0.6) !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-size: 1.05rem !important;
  padding: 0.75rem 1.5rem !important;
  transition: all 0.3s ease !important;
}
.stButton > button:hover {
  background: rgba(63, 169, 245, 0.25) !important;
  border-color: rgba(63, 169, 245, 0.9) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 4px 12px rgba(63, 169, 245, 0.3) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }

.main-wrap{ position:relative; z-index:2; padding:1rem 0 0 0; }
.metric-card{
  background:rgba(20,30,60,.72); border:1px solid rgba(80,120,200,.42);
  border-radius:16px; padding:1.5rem; text-align:center; box-shadow:0 4px 20px rgba(0,0,0,.3);
}
[data-testid="stMetric"] label,[data-testid="stMetricValue"]{ color:#fff !important; }
</style>
""", unsafe_allow_html=True)

# ---------- Background layers ----------
st.markdown("""
<div class="transit-bg" id="transitBg">
  <div class="star"></div>
  <div class="planet"></div>
  <div class="dimmer"></div>
</div>
<div class="global-dimmer" id="globalDimmer"></div>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown('<div class="front-content">', unsafe_allow_html=True)
st.markdown("""
<section class="hero" id="hero">
  <h0 class="space-title">ExoMatch</h0>
  <p class="subtitle">AI × Human collaboration platform for professional and educational exoplanet analysis of the next generation.</p>
  <div class="cta-wrap">
  </div>
            
</section>
""", unsafe_allow_html=True)

# ---------- Body (下方按鈕維持原樣) ----------
st.markdown('<div class="main-wrap" id="mainContent">', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Candidates Loaded", "12,430")
c2.metric("AI-Flagged", "2,310")
c3.metric("Human-Vetted", "7,945")
c4.metric("Confirmed", "124")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="metric-card">
  <h3 style="color:#3fa9f5;margin-top:0;">Our Mission</h3>
  <p style="font-size:1.15rem; color:#ffffff; line-height:1.7;">
    Combining <b>AI precision</b> with <b>human intuition</b> to accelerate the discovery of new exoplanets.
    Collaborate, verify, and contribute to a growing database of confirmed transits.
  </p>
</div>
""", unsafe_allow_html=True)

b1, b2, b3 = st.columns(3)
with b1:
    if st.button("Start Vetting", use_container_width=True, key="btn_start"):
        st.switch_page("pages/vetting.py")
with b2:
    if st.button("Candidate Database", use_container_width=True, key="btn_db"):
        st.switch_page("pages/analyze.py")
with b3:
    if st.button("User Contributions", use_container_width=True, key="btn_user"):
        st.switch_page("pages/about.py")

st.markdown('<br><br><div style="text-align:center;color:#aaa;">ExoMatch v3.8 — Auto Navigation</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)