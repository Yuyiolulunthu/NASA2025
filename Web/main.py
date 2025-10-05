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
  position:absolute; top:50%; left:55%;   /* ← 原本 38%，改 70% 讓星體更靠右 */
  transform:translate(-50%,-50%);
  width:360px;height:360px;border-radius:50%;
  background:radial-gradient(circle at 32% 30%,#fff8e1,#f5b971 70%,#9e6c33 100%);
  box-shadow:0 0 120px 48px rgba(255,215,150,.28);
  z-index:3;transition:filter .18s,box-shadow .18s;
}
.planet{
  position:absolute; top:50%; left:130%;   /* ← 從更右邊出發（原 110%） */
  transform:translateY(-50%) scale(0.9);
  width:72px;height:72px;border-radius:50%;
  background:radial-gradient(circle at 30% 30%,#1a1a1a,#333);
  z-index:4;
  animation:transit 12s linear infinite;
}
@keyframes transit{
  0%{ left:130%; opacity:0; }   /* ← 起點更右 */
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
  z-index:9999; /* 提高到較高層級，確保在上方 */
  padding:2rem;box-sizing:border-box;
  pointer-events: auto; /* 允許互動 */
}

/* 確保所有 hero 子元素可互動（避免上層 pointer-events:none 影響） */
.front-content .hero,
.front-content .hero * ,
.front-content .side-actions,
.front-content .side-actions a {
  pointer-events: auto;
}

/* HERO 內容 */
.hero{max-width:720px;width:100%;text-align:center;
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:.75rem;}
.space-title{margin:0;color:#fff;font-weight:900;font-size:clamp(3rem,8vw,6rem);letter-spacing:.1em;text-shadow:0 2px 6px rgba(0,0,0,.35);}
.subtitle{margin:.3rem 0 0 0;color:#f5f7ff;opacity:.96;font-weight:700;font-size:clamp(1.6rem,3.5vw,2.2rem);text-shadow:0 2px 6px rgba(0,0,0,.30);}

/* 2×2 快捷連結 */
.side-actions{
  display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-top:1.2rem;width:420px;
  z-index:10000;
}
.side-actions a{
  position:relative;
  z-index:10001;
  display:inline-flex;align-items:center;justify-content:center;
  padding:.75rem 1rem;font-weight:800;color:#e6eefc;text-decoration:none;
  background:rgba(20,30,60,0.6);border:1px solid rgba(99,102,241,.14);
  border-radius:10px;transition:background .16s,transform .12s,color .16s,border-color .16s;
}
.side-actions a:hover{background:#e5e7eb;color:#0b1220;transform:translateY(-2px);border-color:rgba(99,102,241,.5);}
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
st.markdown("""
<section class="hero" id="hero">
  <h0 class="space-title">ExoMatch</h0>
  <p class="subtitle">AI × Human collaboration platform for professional and educational exoplanet analysis.</p>
  <nav class="side-actions">
    <a href="/about" target="_self">About our model</a>
    <a href="/analyze" target="_self">Analyze your data</a>
    <a href="/fits_converter" target="_self">FITS Converter</a>
    <a href="/vetting" target="_self">Learning about vetting</a>
  </nav>
</section>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- JS：暗化控制 ----------
html("""
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
""", height=0)
