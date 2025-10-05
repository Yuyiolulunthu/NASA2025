# Web/components/banner.py
import base64
from pathlib import Path
import streamlit as st

PRIMARY_BG = "#151622"  # 深藍黑背景

def _logo_data_uri() -> str:
    for p in ["Web/logo.png", "./Web/logo.png", "logo.png", "./logo.png"]:
        f = Path(p)
        if f.exists():
            try:
                return "data:image/png;base64," + base64.b64encode(f.read_bytes()).decode()
            except Exception:
                pass
    # 後備 SVG（找不到 logo 檔時顯示）
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="34" height="34" viewBox="0 0 64 64">
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
/* 隱藏 Streamlit 側邊欄 */
section[data-testid="stSidebar"], div[data-testid="stSidebar"], [aria-label="Sidebar"] {{ display: none !important; }}

/* 頂部固定 banner 與內容間距 */
div.block-container {{ padding-top: 64px !important; }}
.topnav {{
  position: fixed; top: 0; left: 0; right: 0; height: 60px;
  background: {bg}; color: #fff; z-index: 9999;
  display: flex; align-items: center; border-bottom: 1px solid rgba(102,126,234,.15);
  padding: 0 16px;
}}
.topnav-inner {{ display:flex; align-items:center; justify-content:space-between; width:100%; max-width:1200px; margin:0 auto; }}
.brand {{ display:flex; align-items:center; gap:10px; font-weight:800; font-size:1.05rem; }}
.brand img {{ width:32px; height:32px; display:block; }}
.nav-actions {{ display:flex; gap:10px; align-items:center; }}
.nav-actions a {{
  color: #e5e7eb; text-decoration:none; font-weight:700; padding:8px 10px; border-radius:8px;
  border:1px solid rgba(99,102,241,.12);
}}
.nav-actions a:hover {{ background:#e5e7eb; color:#111827; border-color: rgba(99,102,241,.5); }}

/* 窄螢幕處理：隱藏文字連結，保留 icon (可進一步改為下拉或漢堡) */
@media (max-width:640px) {{
  .nav-actions a {{ padding:8px; font-size:0.95rem; }}
  .brand span {{ display:none; }}
}}
</style>

<div class='topnav' role="navigation" aria-label="Primary">
  <div class='topnav-inner'>
    <div class='brand'>
      <img src="{logo}" alt="logo"/>
      <span>ExoMatch</span>
    </div>
    <div class='nav-actions' role="menu" aria-label="Top links">
      <a href="/">Home</a>
      <a href="/about">About our model</a>
      <a href="/analyze">Analyze your data</a>
      <a href="/fits_converter">FITS Converter</a>
      <a href="/vetting">Learning about vetting</a>
    </div>
  </div>
</div>
    """.format(bg=PRIMARY_BG, logo=logo), unsafe_allow_html=True)