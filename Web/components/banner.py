# Web/components/banner.py
import base64
from pathlib import Path
import streamlit as st

PRIMARY_BG = "#151622"  # 深藍黑背景

def _logo_data_uri() -> str:
    for p in ["Web/logo.png", "./Web/logo.png", "logo.png", "./logo.png"]:
        f = Path(p)
        if f.exists():
            return "data:image/png;base64," + base64.b64encode(f.read_bytes()).decode()
    # 後備 SVG（找不到 logo 檔時顯示）
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 64 64">
      <defs><linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
        <stop offset="0%" stop-color="#4facfe"/><stop offset="100%" stop-color="#667eea"/>
      </linearGradient></defs>
      <circle cx="32" cy="32" r="30" fill="url(#g)"/>
      <text x="32" y="41" text-anchor="middle" font-size="24" font-family="Inter,system-ui,sans-serif" fill="#fff">E</text>
    </svg>
    """.strip()
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def render_banner():
    logo = _logo_data_uri()
    st.markdown("""
<style>
div.block-container {{ padding-top: 84px !important; }}

/* ===== 固定頂部 Banner ===== */
.exo-top {{
  position: fixed; inset: 0 0 auto 0; height: 64px;
  background: {bg};
  border-bottom: 1px solid rgba(102,126,234,.35);
  backdrop-filter: blur(8px);
  z-index: 9999;
}}

/* 中央：Logo + Title（用 button 模擬連結） */
.exo-center-form {{
  position: absolute; left: 50%; top: 50%; transform: translate(-50%,-50%);
  margin: 0;
}}
.exo-center-btn {{
  display: inline-flex; align-items: center; gap: 10px;
  background: transparent; border: none; padding: 0; cursor: pointer;
}}
.exo-center-btn img {{ width: 36px; height: 36px; display: block; }}
.exo-center-btn span {{
  color:#fff; font-weight: 900; letter-spacing:.03em; font-size: 1.55rem; line-height: 1;
}}

/* 右上三槓：預設黑底白槓；hover 白底黑槓 */
#exo-toggle {{ display: none; }}
.exo-btn {{
  position:absolute; right:18px; top:50%; transform: translateY(-50%);
  width:44px; height:36px; display:flex; align-items:center; justify-content:center;
  border:1px solid rgba(99,102,241,.35);
  border-radius:10px; cursor:pointer;
  background:{bg}; color:#e5e7eb;
  transition: background .2s,color .2s,border-color .2s;
}}
.exo-btn:hover {{ background:#e5e7eb; color:#111827; border-color:rgba(99,102,241,.6); }}

.exo-bars, .exo-bars::before, .exo-bars::after {{
  content:""; display:block; width:22px; height:2px; background: currentColor;
  position: relative; border-radius:2px;
}}
.exo-bars::before {{ position:absolute; top:-6px; }}
.exo-bars::after  {{ position:absolute; top: 6px; }}

/* ===== 簡潔下拉（由上往下展開） ===== */
.exo-menu {{
  position: fixed; right: 18px; top: 64px;
  max-height: 0; overflow: hidden; padding: 0;
  transition: max-height .28s ease, padding .28s ease;
  z-index: 9998;
}}
#exo-toggle:checked ~ .exo-menu {{ max-height: 260px; padding: 6px 0; }}

.exo-menu ul {{ list-style: none; margin: 0; padding: 0; text-align: left; }}

/* 用 form+button 當連結，確保同分頁導向 */
.exo-link-form {{ margin: 0; }}
.exo-link-btn {{
  display: block; width: 100%; text-align: left;
  background: transparent; border: none; padding: 6px 0; cursor: pointer;
  color: #e5e7eb; font-weight: 800; line-height: 1.2;
}}
.exo-link-btn:hover {{ color: #9aa7ff; }}

/* demo 條目縮排 + 小字 */
.exo-subitem .exo-link-btn {{ margin-left: 14px; font-size: .92rem; font-weight: 700; }}
</style>

<div class="exo-top">
  <!-- 中央：點擊回首頁（不開新分頁） -->
  <form class="exo-center-form" method="get" action="/">
    <button type="submit" class="exo-center-btn" title="Back to main">
      <img src="{logo}" alt="ExoMatch logo" />
      <span>ExoMatch</span>
    </button>
  </form>

  <!-- 右上三槓 -->
  <input type="checkbox" id="exo-toggle"/>
  <label for="exo-toggle" class="exo-btn" title="More">
    <span class="exo-bars"></span>
  </label>

  <!-- 下拉選單（同分頁導向的表單按鈕） -->
  <nav class="exo-menu">
    <ul>
      <li>
        <form class="exo-link-form" method="get" action="/about">
          <button class="exo-link-btn" type="submit">About our model</button>
        </form>
      </li>
      <li>
        <form class="exo-link-form" method="get" action="/analyze">
          <button class="exo-link-btn" type="submit">Analyze your data</button>
        </form>
      </li>
      <li class="exo-subitem">
        <form class="exo-link-form" method="get" action="/analyze-demo">
          <button class="exo-link-btn" type="submit">Try our analysis demo</button>
        </form>
      </li>
      <li>
        <form class="exo-link-form" method="get" action="/vetting">
          <button class="exo-link-btn" type="submit">Vet your data</button>
        </form>
      </li>
      <li class="exo-subitem">
        <form class="exo-link-form" method="get" action="/vetting-demo">
          <button class="exo-link-btn" type="submit">Try our vetting demo</button>
        </form>
      </li>
    </ul>
  </nav>
</div>
    """.format(bg=PRIMARY_BG, logo=logo), unsafe_allow_html=True)
