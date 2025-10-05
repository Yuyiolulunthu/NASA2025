import streamlit as st
from components.banner import render_banner
from streamlit.components.v1 import html

# ---------- Page Config ----------
st.set_page_config(
    page_title="ExoMatch - Exoplanet Hunter",
    page_icon="Web/logo.png",
    layout="wide",
)

# ---------- Banner & 基本隱藏 ----------
render_banner()
st.markdown("""
    <style>
    #MainMenu, footer, header {visibility:hidden;}
    </style>
""", unsafe_allow_html=True)

# ---------- CSS + 靜態 HTML ----------
st.markdown("""
<style>
/* ===== 基礎設定 ===== */
[data-testid="stAppViewContainer"]{
  background: radial-gradient(circle at 50% 50%, #0b0f1a 0%, #0e162a 60%, #0a0e27 100%);
  color:#eaeef7; font-family:'Inter','Open Sans',sans-serif; overflow-x:hidden;
}

/* ===== 背景動畫（限定於左半邊） ===== */
.transit-bg{
  position:fixed; top:0; bottom:0; left:0;
  width:50vw; z-index:0; pointer-events:none; overflow:hidden; display:block;
  filter:brightness(1); transition:filter .15s linear;
  opacity:0; transform:translateX(-6%);
  animation:transitFadeIn 900ms cubic-bezier(.2,.9,.2,1) forwards;
  animation-delay:200ms;
}
.transit-bg::after{
  content:""; position:absolute; top:0; bottom:0; right:0; width:8vw;
  pointer-events:none; background: linear-gradient(to right, rgba(10,12,20,0), rgba(10,12,20,0.65));
  z-index:1;
}
@keyframes transitFadeIn{ to { opacity:1; transform:translateX(0); } }

/* 大球（star） */
.star{
  position:absolute; top:50%; left:38%;
  transform:translate(-50%,-50%) scale(1);
  width:360px; height:360px; border-radius:50%;
  background:radial-gradient(circle at 32% 30%, #fff8e1, #f5b971 70%, #9e6c33 100%);
  box-shadow:0 0 120px 48px rgba(255,215,150,.28);
  z-index:3;
  transition:filter .18s ease, box-shadow .18s ease, transform .18s ease;
  transform-origin:center;
}
/* 強制暗化類別（JS 加/移除） */
.star.is-dim{
  filter:brightness(0.62) !important;
  box-shadow:0 0 44px 12px rgba(0,0,0,.5) !important;
}

/* 小球（planet）：等速移動（linear） */
.planet{
  position:absolute; top:50%; left:110%;
  transform:translateY(-50%) scale(0.9);
  width:72px; height:72px; border-radius:50%;
  background:radial-gradient(circle at 30% 30%, #1a1a1a, #333);
  z-index:4;
  will-change:left,transform,opacity,filter;
  animation:transit 12s linear infinite;
  transition:filter .12s linear, transform .12s linear, opacity .12s linear;
  filter:brightness(1);
}
/* planet 暗化類別 */
.planet.is-dim{
  filter:brightness(0.68) !important;
}
@keyframes transit{
  0%{ left:110%; opacity:0; transform:translateY(-50%) scale(0.9); }
  6%{ opacity:1; }
  94%{ opacity:1; }
  100%{ left:-10%; opacity:0; transform:translateY(-50%) scale(0.9); }
}

/* dimmer（整頁微暗效果） */
.dimmer{
  position:fixed; inset:0; z-index:2; pointer-events:none;
  background:rgba(0,0,0,0); animation:dimmer 12s linear infinite;
}
@keyframes dimmer{
  0%,38%{background:rgba(0,0,0,0);}
  45%,55%{background:rgba(0,0,0,.12);}
  62%,100%{background:rgba(0,0,0,0);}
}

/* ===== 全頁暗化層（預留） ===== */
.global-dimmer{
  position:fixed; inset:0; z-index:100000; pointer-events:none;
  background:rgba(0,0,0,0); transition:background .16s ease;
}

/* ===== 排版（右半 Hero 置中） ===== */
:root { --banner-height: 64px; }

/* front-content 固定於右半邊並垂直置中 */
.front-content{
  position: fixed;
  top: var(--banner-height);
  right: 0;
  width: 50vw;                        /* 右半邊 */
  height: calc(100vh - var(--banner-height));
  display: flex;
  align-items: center;
  justify-content: center;
  box-sizing: border-box;
  padding: 2rem;
  z-index: 2;                         /* 確保在動態背景上方 */
  pointer-events: auto;
}

/* 行動裝置回到單欄排列 */
@media (max-width: 900px) {
  .front-content{
    position: static;
    width: 100%;
    height: auto;
    padding: 2rem 1rem;
  }
  .transit-bg{ display:none; } /* 行動裝置可選擇隱藏左側動畫 */
}

/* ===== HERO（內容寬度限制） ===== */
.hero{ max-width: 720px; width: 100%; text-align: center; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:0; padding-inline:2rem; }
.space-title{ margin:0; color:#fff; font-weight:900; font-size:clamp(3rem,8vw,6rem); letter-spacing:.1em; text-shadow:0 2px 6px rgba(0,0,0,.35); }
.subtitle{ margin:.3rem 0 0 0; color:#f5f7ff; opacity:.96; font-weight:700; font-size:clamp(1.6rem,3.5vw,2.6rem); text-shadow:0 2px 6px rgba(0,0,0,.30); }

/* ===== CTA（保留樣式，未放按鈕） ===== */
.cta-wrap{ margin-top: clamp(6vh, 8vh, 10vh); display:flex; justify-content:center; align-items:center; }
.cta-button{
  display:inline-flex; align-items:center; justify-content:center;
  width:270px; height:78px; background:transparent; color:#ffffff;
  border:2px solid rgba(235,238,245,0.9); border-radius:0;
  font-weight:900; font-size:1.3rem; letter-spacing:.06em; cursor:pointer;
  transition: transform .22s ease, background .22s ease, color .22s ease, border-color .22s ease;
}
.cta-button:hover{
  background:#ffffff; color:#0c1225; border-color:#ffffff; transform:translateY(-2px);
}

/* ===== 下方一般按鈕樣式 ===== */
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

<!-- 左側動畫：星體 + 行星 + 微暗層 -->
<div class="transit-bg" id="transitBg">
  <div class="star" aria-hidden="true"></div>
  <div class="planet" aria-hidden="true"></div>
  <div class="dimmer"></div>
</div>
<div class="global-dimmer" id="globalDimmer"></div>
""", unsafe_allow_html=True)

# ---------- HERO（右半置中） ----------
st.markdown('<div class="front-content">', unsafe_allow_html=True)
st.markdown("""
<section class="hero" id="hero">
  <h0 class="space-title">ExoMatch</h0>
  <div class="cta-wrap"></div>
</section>
""", unsafe_allow_html=True)
# 關閉 front-content，避免影響下方主內容
st.markdown('</div>', unsafe_allow_html=True)

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
# 關閉 main-wrap
st.markdown('</div>', unsafe_allow_html=True)

# ---------- JS：用 iframe 執行，操控父頁面元素（暗化邏輯） ----------
# NOTE: 若 CSP 導致無法取 parent，程式會退回同文件查找（保險）
html("""
<script>
(function(){
  const THRESHOLD = 200; // px (可依需求調整)
  let star = null, planet = null;

  // 嘗試取父頁面 document，否則退回當前 document
  let PARENT_DOC = null;
  try {
    PARENT_DOC = window.parent && window.parent.document ? window.parent.document : document;
  } catch (e) {
    PARENT_DOC = document;
  }

  function ready() {
    star = PARENT_DOC.querySelector('.star');
    planet = PARENT_DOC.querySelector('.planet');
    if (!star || !planet) { setTimeout(ready, 150); return; }
    requestAnimationFrame(loop);
  }

  function loop(){
    if (!star || !planet){ requestAnimationFrame(loop); return; }
    const sr = star.getBoundingClientRect();
    const pr = planet.getBoundingClientRect();

    const sx = sr.left + sr.width / 2;
    const sy = sr.top + sr.height / 2;
    const px = pr.left + pr.width / 2;
    const py = pr.top + pr.height / 2;

    const centerDist = Math.hypot(sx - px, sy - py);

    if (centerDist <= THRESHOLD) {
      star.classList.add('is-dim');
      planet.classList.add('is-dim');
    } else {
      star.classList.remove('is-dim');
      planet.classList.remove('is-dim');
    }
    requestAnimationFrame(loop);
  }

  if (PARENT_DOC.readyState === 'complete') {
    ready();
  } else {
    try { window.parent.addEventListener('load', ready); }
    catch(e){ window.addEventListener('load', ready); }
  }
})();
</script>
""", height=0)
