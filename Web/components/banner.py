# components/banner.py
import base64
from pathlib import Path
import streamlit as st


def _get_logo_data_uri() -> str:
    """載入 logo 圖片並轉成 base64"""
    for p in ["Web/logo.png", "./Web/logo.png", "logo.png", "./logo.png"]:
        f = Path(p)
        if f.exists():
            try:
                b = base64.b64encode(f.read_bytes()).decode()
                return f"data:image/png;base64,{b}"
            except Exception:
                pass

    # fallback SVG
    fallback_svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 64 64">
      <circle cx="32" cy="32" r="30" fill="#667eea"/>
      <text x="32" y="40" font-size="24" text-anchor="middle" fill="white" font-family="sans-serif">E</text>
    </svg>
    """.strip()
    return "data:image/svg+xml;base64," + base64.b64encode(fallback_svg.encode()).decode()


def render_banner():
    logo_src = _get_logo_data_uri()

    st.markdown(f"""
<style>
/* =================== Base =================== */
div.block-container {{ padding-top: 84px !important; }}

/* =================== Banner =================== */
.exo-top-banner {{
  position: fixed;
  top: 0; left: 0; right: 0;
  height: 64px;
  background: #151622;   /* ← 換成你的深色 */
  border-bottom: 1px solid rgba(102,126,234,0.3);
  backdrop-filter: blur(8px);
  z-index: 9999;
}}

/* =================== Center Logo =================== */
.exo-center {{
  position: absolute;
  left: 50%; top: 50%;
  transform: translate(-50%, -50%);
  display: flex; align-items: center; gap: 10px;
  cursor: pointer; text-decoration: none;
}}
.exo-center img {{ width: 36px; height: 36px; }}
.exo-center span {{
  color: white;
  font-size: 1.25rem;
  font-weight: 800;
  letter-spacing: .05em;
}}

/* =================== Right Button =================== */
.exo-menu-btn {{
  position: absolute;
  right: 20px; top: 50%;
  transform: translateY(-50%);
  width: 44px; height: 36px;
  display: flex; align-items: center; justify-content: center;
  border: 1px solid rgba(99,102,241,.35);
  border-radius: 10px;
  background: #e5e7eb; color: #111827;
  transition: background .2s, color .2s, border-color .2s;
  cursor: pointer;
}}
.exo-menu-btn:hover {{
  background: #151622;
  color: #e5e7eb;
  border-color: rgba(99,102,241,.6);
}}
.exo-menu-btn div,
.exo-menu-btn div::before,
.exo-menu-btn div::after {{
  content:""; display:block;
  width:22px; height:2px; background:currentColor;
  position:relative; border-radius:2px;
}}
.exo-menu-btn div::before {{ position:absolute; top:-6px; }}
.exo-menu-btn div::after  {{ position:absolute; top: 6px; }}

/* =================== Slide Menu =================== */
.exo-menu {{
  position: fixed;
  top: 64px; right: -280px;
  width: 260px; height: calc(100vh - 64px);
  background: #151622;   /* ← 同樣換色 */
  border-left: 1px solid rgba(102,126,234,0.3);
  padding: 1.5rem;
  transition: right .3s ease;
  z-index: 9998;
}}
.exo-menu.show {{ right: 0; }}

.exo-item {{
  display:block;
  color:#e5e7eb;
  border:1px solid rgba(99,102,241,.3);
  border-radius:8px;
  padding:.6rem .8rem;
  margin-bottom:.8rem;
  font-weight:600;
  text-decoration:none;
  transition: background .2s, color .2s, border-color .2s;
}}
.exo-item:hover {{
  background:#e5e7eb;
  color:#151622;
  border-color:rgba(99,102,241,.6);
}}
.exo-sub {{
  margin-left:.5rem;
  padding-left:.8rem;
  border-left:1px dashed rgba(148,163,184,.35);
}}
</style>

<!-- =================== Layout =================== -->
<div class="exo-top-banner">
  <a href="?page=main" class="exo-center">
    <img src="{logo_src}" alt="logo">
    <span>ExoMatch</span>
  </a>

  <div class="exo-menu-btn" id="menu-btn">
    <div></div>
  </div>
</div>

<div class="exo-menu" id="menu-panel">
  <a href="?page=about" class="exo-item">About our model</a>
  <a href="?page=analyze" class="exo-item">Analyze your data</a>
  <div class="exo-sub">
    <a href="?page=analyze_demo" class="exo-item">Try our analysis demo</a>
  </div>
  <a href="?page=vetting" class="exo-item">Vet your data</a>
  <div class="exo-sub">
    <a href="?page=vetting_demo" class="exo-item">Try our vetting demo</a>
  </div>
</div>

<!-- =================== JS (內嵌版確保可執行) =================== -->
<script>
setTimeout(() => {{
  const btn = document.querySelector("#menu-btn");
  const panel = document.querySelector("#menu-panel");
  if (!btn || !panel) return;
  btn.addEventListener("click", () => {{
    panel.classList.toggle("show");
  }});
  document.addEventListener("click", (e) => {{
    if (!panel.classList.contains("show")) return;
    const insidePanel = panel.contains(e.target);
    const insideBtn = btn.contains(e.target);
    if (!insidePanel && !insideBtn) {{
      panel.classList.remove("show");
    }}
  }});
}}, 500);
</script>
    """, unsafe_allow_html=True)
