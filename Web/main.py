import streamlit as st
from components.banner import render_banner

# ---------- Page Config ----------
st.set_page_config(
    page_title="Exoplanet Hunter",
    page_icon="🌌",
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
/* ===== Globals ===== */
[data-testid="stAppViewContainer"]{
  background: radial-gradient(circle at 50% 50%, #0b0f1a 0%, #0e162a 60%, #0a0e27 100%);
  color:#eaeef7; font-family:'Inter','Open Sans',sans-serif; overflow-x:hidden;
}

/* ===== Background (star + planet) ===== */
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
/* Planetary transit dim pulse (optional aesthetic) */
.dimmer{ position:fixed; inset:0; z-index:1; pointer-events:none; background:rgba(0,0,0,0); animation:dimmer 12s linear infinite; }
@keyframes dimmer{ 0%,38%{background:rgba(0,0,0,0);} 45%,55%{background:rgba(0,0,0,.2);} 62%,100%{background:rgba(0,0,0,0);} }

/* ===== Global scroll dimmer (covers EVERYTHING) ===== */
.global-dimmer{
  position:fixed; inset:0; z-index:100000; pointer-events:none;
  background:rgba(0,0,0,0); transition:background .16s ease;
}

/* ===== Foreground content ===== */
.front-content{ position:relative; z-index:2; }

/* ===== HERO (微下移 + 置中) ===== */
.hero{
  min-height:100vh;
  display:flex; flex-direction:column; align-items:center; text-align:center;
  justify-content:flex-start;
  padding-top:clamp(22vh, 24vh, 28vh);     /* ⬅ 再下移一點點（原來 20~26vh） */
  padding-inline:2rem; gap:1rem;
}
.hero-inner{ transform-origin:center top; transition:transform .06s linear; }

/* Title + Subtitle (極淡黑色光暈，提升可讀性但不明顯) */
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

/* CTA 包層：方框再往下 */
.cta-wrap{ margin-top:clamp(18vh, 14vh, 18vh); }  /* ⬅ 再下移（原來 10~15vh） */

/* CTA 方框（半透明、硬邊） */
.square-btn{
  display:inline-flex; align-items:center; justify-content:center;
  width:260px; height:66px;
  font-weight:900; font-size:1.15rem; letter-spacing:.08em;
  color:#fff; background:rgba(0,0,0,.18);
  border:2px solid rgba(235,238,245,.95); border-radius:0;
  backdrop-filter:blur(.5px);
  transition:transform .22s ease, background .22s ease, color .22s ease, border-color .22s ease;
  user-select:none; cursor:pointer;
}
.square-btn:hover{ background:#fff; color:#0c1225; border-color:#fff; transform:translateY(-2px); }

/* ===== Main section ===== */
.main-wrap{ position:relative; z-index:2; padding:1rem 0 0 0; }
.metric-card{
  background:rgba(20,30,60,.72); border:1px solid rgba(80,120,200,.42);
  border-radius:16px; padding:1.5rem; text-align:center; box-shadow:0 4px 20px rgba(0,0,0,.3);
}
[data-testid="stMetric"] label,[data-testid="stMetricValue"]{ color:#fff !important; }
</style>
""", unsafe_allow_html=True)

# ---------- Background layers & Global dimmer ----------
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
  <div class="hero-inner" id="heroInner">
    <h1 class="space-title">EXOPLANET HUNTER</h1>
    <p class="subtitle">AI × Human Collaboration Platform for Professional Exoplanet Vetting</p>
    <div class="cta-wrap">
      <div class="square-btn" id="startAnalyzingBtn">&gt; Start Analyzing</div>
    </div>
  </div>
</section>
""", unsafe_allow_html=True)

# ---------- Body ----------
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
with b1: st.button("Start Vetting", use_container_width=True, key="btn_start")
with b2: st.button("Candidate Database", use_container_width=True, key="btn_db")
with b3: st.button("User Contributions", use_container_width=True, key="btn_user")

st.markdown('<br><br><div style="text-align:center;color:#aaa;">Exoplanet Hunter v3.7 — Micro Downshift + Guaranteed Fade</div>', unsafe_allow_html=True)
st.markdown('</div></div>', unsafe_allow_html=True)

# ---------- JS (robust: direct style changes, no CSS vars required) ----------
st.markdown("""
<script>
(function(){
  // Robust DOM grab with fallback/polling
  function $(sel){ return document.querySelector(sel); }
  function onReady(fn){
    if (document.readyState === 'complete' || document.readyState === 'interactive') fn();
    else document.addEventListener('DOMContentLoaded', fn);
  }

  onReady(function(){
    const hero = document.getElementById('hero');
    const main = document.getElementById('mainContent');
    const startBtn = document.getElementById('startAnalyzingBtn');
    const globalDimmer = document.getElementById('globalDimmer');
    const transitBg   = document.getElementById('transitBg');

    // In case any element missing, bail gracefully
    if (!globalDimmer || !transitBg) return;

    // Smooth scroll
    if (startBtn && main){
      startBtn.addEventListener('click', function(){
        main.scrollIntoView({behavior:'smooth', block:'start'});
      });
    }

    const DIM_MAX = 0.85;      // 全頁最暗不透明度（越大越暗）
    const BG_MIN  = 0.35;      // 背景最小亮度（越小越暗）

    function clamp(v,a,b){ return Math.min(b, Math.max(a, v)); }

    function update(){
      const heroH = Math.max(300, (hero ? hero.offsetHeight : window.innerHeight));
      const y = window.scrollY || window.pageYOffset || 0;

      // 區間：開始淡化到完全淡化
      const start = heroH * 0.08;
      const end   = heroH * 0.65;
      let t = (y - start) / (end - start);
      t = clamp(t, 0, 1);

      // 1) 全頁暗化：直接改遮罩背景，保證有效
      const alpha = (DIM_MAX * t).toFixed(3);
      globalDimmer.style.background = 'rgba(0,0,0,' + alpha + ')';

      // 2) 背景亮度：直接改 filter:brightness，保證有效
      const bright = (1 - (1 - BG_MIN) * t).toFixed(3);
      transitBg.style.filter = 'brightness(' + bright + ')';
    }

    window.addEventListener('scroll', update, {passive:true});
    window.addEventListener('resize', update, {passive:true});
    update(); // initial
  });
})();
</script>
""", unsafe_allow_html=True)
