import streamlit as st
from components.banner import render_banner
from streamlit.components.v1 import html

# ---------- Page Config ----------
st.set_page_config(
    page_title="ExoMatch - Exoplanet Hunter",
    page_icon="Web/logo.png",
    layout="wide",
)

# ---------- Banner & 隱藏 ----------
render_banner()
st.markdown("""
<style>
#MainMenu, footer, header {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- CSS + 靜態 HTML ----------
st.markdown("""
<style>
/* ===== 背景基礎 ===== */
[data-testid="stAppViewContainer"]{
  background: radial-gradient(circle at 50% 50%, #0b0f1a 0%, #0e162a 60%, #0a0e27 100%);
  color:#eaeef7; font-family:'Inter','Open Sans',sans-serif; overflow-x:hidden;
}

/* ===== 右半動畫區 ===== */
.transit-bg{
  position:fixed; top:0; bottom:0; right:0;
  width:50vw; z-index:0; pointer-events:none; overflow:hidden;
  filter:brightness(1); transition:filter .15s linear;
  opacity:0; transform:translateX(6%);
  animation:fadeIn 900ms cubic-bezier(.2,.9,.2,1) forwards;
  animation-delay:200ms;
}
.transit-bg::after{
  content:""; position:absolute; top:0; bottom:0; left:0; width:8vw;
  background:linear-gradient(to left, rgba(10,12,20,0), rgba(10,12,20,0.65));
}
@keyframes fadeIn{to{opacity:1;transform:translateX(0);}}

/* 星體與行星 */
.star{
  position:absolute; top:50%; left:55%; /* 將星體偏右，保留視覺重心 */
  transform:translate(-50%,-50%);
  width:360px;height:360px;border-radius:50%;
  background:radial-gradient(circle at 32% 30%,#fff8e1,#f5b971 70%,#9e6c33 100%);
  box-shadow:0 0 120px 48px rgba(255,215,150,.28);
  z-index:3;transition:filter .18s,box-shadow .18s;
}
.planet{
  position:absolute; top:50%; left:130%;
  transform:translateY(-50%) scale(0.9);
  width:72px;height:72px;border-radius:50%;
  background:radial-gradient(circle at 30% 30%,#1a1a1a,#333);
  z-index:4;
  animation:transit 12s linear infinite;
}
@keyframes transit{
  0%{ left:130%; opacity:0; }
  6%{ opacity:1; }
  94%{ opacity:1; }
  100%{ left:-10%; opacity:0; }
}

/* 暗化效果 */
.star.is-dim{filter:brightness(0.62)!important;box-shadow:0 0 44px 12px rgba(0,0,0,.5)!important;}
.planet.is-dim{filter:brightness(0.68)!important;}

/* ===== 左半 HERO ===== */
:root {--banner-height:64px;}

.front-content{
  position:fixed;top:var(--banner-height);
  left:0;right:50vw;
  width:auto;height:calc(100vh - var(--banner-height));
  display:flex;align-items:center;justify-content:center;
  z-index:9999; /* 確保在上方 */
  padding:2rem;box-sizing:border-box;
  pointer-events: auto; /* 允許互動 */
}

/* 確保所有 hero 子元素可互動 */
.front-content .hero,
.front-content .hero *,
.front-content .side-actions,
.front-content .side-actions a { pointer-events: auto; }

/* HERO 內容 */
.hero{max-width:760px;width:100%;text-align:center;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1rem;}
.space-title{margin:0;color:#fff;font-weight:900;font-size:clamp(6.5rem, 16.4vw, 12.5rem);letter-spacing:.06em;text-shadow:0 2px 6px rgba(0,0,0,.35);}
.subtitle{margin:.25rem 0 0 0;color:#f5f7ff;opacity:.96;font-weight:700;font-size:clamp(1.35rem,3.2vw,2.05rem);text-shadow:0 2px 6px rgba(0,0,0,.30);} 
.lead{margin:.25rem 0 0 0;color:#d9e1ff;opacity:.9;font-weight:500;font-size:clamp(0.95rem,1.7vw,1.05rem);} 

/* 2×2 導覽卡片（含引導文字） */
.side-actions{
  display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-top:1.2rem;width:760px;max-width:100%;
  z-index:10000;
}
.side-actions a{
  position:relative; display:flex; flex-direction:column; align-items:flex-start; justify-content:flex-start; text-align:left;
  padding:1rem 1.1rem 1.05rem 1.1rem; gap:.35rem; width:100%; min-height:112px;
  font-weight:700; color:#e6eefc; text-decoration:none;
  background:linear-gradient(180deg, rgba(20,30,60,0.65) 0%, rgba(18,28,52,0.50) 100%);
  border:1px solid rgba(129,140,248,.18);
  border-radius:10px; transition:background .16s,transform .12s,color .16s,border-color .16s, box-shadow .16s;
}
.side-actions a:focus-visible{outline:2px solid #7aa2ff; outline-offset:2px; border-radius:12px;}
.side-actions a:hover{
  background:rgba(229,231,235,0.96); color:#0b1220; transform:translateY(-2px);
  border-color:rgba(99,102,241,.55); box-shadow:0 10px 30px rgba(0,0,0,.30);
}
.card-title{font-size:1.75rem; line-height:1.25; letter-spacing:.03em; font-weight:800;}
.card-desc{font-size:.92rem; line-height:1.35; font-weight:550; opacity:.88;}

/* 小尺寸時改為單欄 */
@media(max-width:900px){.side-actions{width:100%;grid-template-columns:1fr;}}
</style>

<!-- 右半動畫 -->
<div class="transit-bg">
  <div class="star"></div>
  <div class="planet"></div>
  <div class="dimmer"></div>
</div>
""", unsafe_allow_html=True)

# ---------- HERO (左半) ----------
st.markdown('<div class="front-content">', unsafe_allow_html=True)
st.markdown(
    """
<section class="hero" id="hero">
  <h1 class="space-title">ExoMatch</h1>
  <p class="subtitle">AI × Human collaboration platform for professional exoplanet analysis</p>
  <p class="lead">Start with a clear path. Choose one of the options below based on what you want to do today.</p>
  <nav class="side-actions" aria-label="Primary">
    <a href="/about" target="_self" aria-label="About our model">
      <span class="card-title">About our model</span>
      <span class="card-desc">How we approach ~96% validated accuracy: rigorous cross-validation, mission-to-mission generalization, and ablation-backed design choices.</span>
    </a>
    <a href="/analyze" target="_self" aria-label="Analyze your data">
      <span class="card-title">Analyze your data</span>
      <span class="card-desc">Upload TESS/Kepler light curves (single or batch). We denoise, extract features, and classify — whether you’re hunting a new candidate or taming a large dataset.</span>
    </a>
    <a href="/fits_converter" target="_self" aria-label="FITS Converter">
      <span class="card-title">FITS Converter</span>
      <span class="card-desc">Turn FITS into calibrated light curves and visual diagnostics in seconds. ESA-aligned tools integrated with our prediction model to surface all target context on one screen.</span>
    </a>
    <a href="/vetting" target="_self" aria-label="Learning about vetting">
      <span class="card-title">Learning about vetting</span>
      <span class="card-desc">New to vetting or curious about the ML pipeline? Explore real cases, decision criteria, and best practices to build expert-level intuition.</span>
    </a>
  </nav>
</section>
""",
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- JS：暗化控制 ----------
html(
    """
<script>
(function(){
  const THRESHOLD = 200; // px
  let star=null,planet=null;
  let doc;
  try{doc=window.parent?.document||document;}catch(e){doc=document;}

  function ready(){
    star=doc.querySelector('.star');
    planet=doc.querySelector('.planet');
    if(!star||!planet){setTimeout(ready,150);return;}
    requestAnimationFrame(loop);
  }

  function loop(){
    if(!star||!planet){requestAnimationFrame(loop);return;}
    const s=star.getBoundingClientRect(), p=planet.getBoundingClientRect();
    const sx=s.left+s.width/2, sy=s.top+s.height/2;
    const px=p.left+p.width/2, py=p.top+p.height/2;
    const d=Math.hypot(sx-px,sy-py);
    if(d<=THRESHOLD){star.classList.add('is-dim');planet.classList.add('is-dim');}
    else{star.classList.remove('is-dim');planet.classList.remove('is-dim');}
    requestAnimationFrame(loop);
  }

  if(doc.readyState==='complete') ready();
  else (window.parent||window).addEventListener('load',ready);
})();
</script>
""",
    height=0,
)
