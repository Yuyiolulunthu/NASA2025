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
/* 內容與固定 Banner 的間距：更緊實（64px） */
div.block-container {{ padding-top: 64px !important; }}

/* ====== 固定頂部 Banner（60px 高） ====== */
.exo-top {{
  position: fixed; inset: 0 0 auto 0; height: 60px;
  background: {bg};
  border-bottom: 1px solid rgba(102,126,234,.35);
  backdrop-filter: blur(8px);
  z-index: 9999;
}}

/* 左側 Home 按鈕（新增樣式） */
.exo-left-form {{
  position: absolute; left: 18px; top: 50%; transform: translateY(-50%);
  margin: 0;
}}
.exo-left-btn {{
  display: inline-flex; align-items: center; justify-content: center;
  width: 44px; height: 34px; border-radius: 10px;
  border: 1px solid rgba(99, 102, 241, .18);
  background: transparent; color: #e5e7eb;
  cursor: pointer; padding: 0;
  transition: background .18s, color .18s, border-color .18s;
}}
.exo-left-btn:hover {{ background: #e5e7eb; color: #111827; border-color: rgba(99,102,241,.6); }}
.exo-left-btn:focus-visible {{ outline: 2px solid #9aa7ff; outline-offset: 2px; }}
.exo-home-ico {{ width: 18px; height: 18px; display: block; }}

/* 中央：Logo + Title（用 form+button，避免新分頁） */
.exo-center-form {{
  position: absolute; left: 50%; top: 50%; transform: translate(-50%,-50%);
  margin: 0;
}}
.exo-center-btn {{
  display: inline-flex; align-items: center; gap: 8px;
  background: transparent; border: none; padding: 0; cursor: pointer;
}}
.exo-center-btn img {{ width: 32px; height: 32px; display: block; }}
.exo-center-btn span {{
  color:#fff; font-weight: 900; letter-spacing:.02em;
  font-size: 1.42rem; line-height: 1;
  transform: translateY(1px);
}}

/* 右上三槓：預設黑底白槓；hover 白底黑槓（顏色互換） */
#exo-toggle {{ display: none; }}
.exo-btn {{
  position: absolute; right: 18px; top: 50%; transform: translateY(-50%);
  width: 44px; height: 34px; display: flex; align-items: center; justify-content: center;
  border: 1px solid rgba(99, 102, 241, .35);
  border-radius: 10px; cursor: pointer;
  background: {bg}; color: #e5e7eb;
  transition: background .2s, color .2s, border-color .2s;
}}
.exo-btn:hover {{ background: #e5e7eb; color: #111827; border-color: rgba(99, 102, 241, .6); }}
.exo-btn:focus-visible {{ outline: 2px solid #9aa7ff; outline-offset: 2px; }}

.exo-bars, .exo-bars::before, .exo-bars::after {{
  content: ""; display: block; width: 22px; height: 2px; background: currentColor;
  position: relative; border-radius: 2px;
}}
.exo-bars::before {{ position: absolute; top: -6px; }}
.exo-bars::after  {{ position: absolute; top: 6px; }}

/* ====== 下拉選單（帶背景框一起滑下來） ====== */
.exo-menu {{
  position: fixed;
  right: 18px;
  top: 60px; /* 緊貼 60px 高的 banner */
  background: rgba(21,22,34,0.96);
  border: 1px solid rgba(102,126,234,.35);
  border-radius: 12px;
  box-shadow: 0 12px 30px rgba(2,6,23,.5);

  max-height: 0; overflow: hidden; padding: 0;
  opacity: 0; transform: translateY(-8px);
  transition:
    max-height .28s ease,
    padding .28s ease,
    opacity .24s ease,
    transform .24s ease;
  z-index: 9998;
}}
#exo-toggle:checked ~ .exo-menu {{
  max-height: 420px;
  padding: 10px 14px;
  opacity: 1;
  transform: translateY(0);
}}

.exo-menu ul {{ list-style: none; margin: 0; padding: 0; text-align: left; }}

.exo-link-form {{ margin: 0; }}
.exo-link-btn {{
  display: flex; align-items: left; gap: 8px;
  width: 100%; text-align: left;
  background: transparent; border: none; padding: 6px 4px; cursor: pointer;
  color: #e5e7eb; font-weight: 800; line-height: 1.2;
}}
.exo-link-btn:hover {{ color: #9aa7ff; }}
.exo-link-btn:focus-visible {{ outline: 2px solid #9aa7ff; border-radius: 6px; }}

.exo-ico {{
  width: 16px; height: 16px; display: inline-block; line-height: 0; flex-shrink: 0;
  opacity: .92;
}}

/* demo 條目縮排 + 小字 */
.exo-subitem .exo-link-btn {{ margin-left: 14px; font-size: .92rem; font-weight: 700; }}

/* ====== 目前頁面高亮（由 JS 加 .is-active） ====== */
.exo-menu .is-active .exo-link-btn {{ color:#b9c3ff; }}
.exo-menu .is-active .exo-link-btn .exo-ico {{ opacity: 1; }}
.exo-menu .is-active .exo-link-btn::before {{
  content:""; display:inline-block; width:6px; height:6px; border-radius:999px;
  background:#9aa7ff; margin-right:8px;
}}

/* 外點擊關閉的遮罩（CSS fallback） */
#exo-toggle:checked ~ .exo-overlay {{
  content:""; position: fixed; inset: 60px 0 0 0; /* banner 下方 */
  background: transparent; z-index: 9997;
}}

/* 動畫/可達性：若使用者偏好減少動態，則關閉過渡 */
@media (prefers-reduced-motion: reduce) {{
  .exo-btn, .exo-menu {{ transition: none !important; }}
}}
</style>

<div class="exo-top" role="navigation" aria-label="Primary">
  <!-- 左側 Home 按鈕 -->
  <form class="exo-left-form" method="get" action="/" aria-label="Home">
    <button type="submit" class="exo-left-btn" title="Home" aria-label="Home">
      <svg class="exo-home-ico" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false">
        <path d="M3 11.5L12 4l9 7.5V20a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1v-8.5z" fill="currentColor"/>
      </svg>
    </button>
  </form>

  <!-- 中央：點擊回首頁（不開新分頁） -->
  <form class="exo-center-form" method="get" action="/" aria-label="Go to Home">
    <button type="submit" class="exo-center-btn" title="Back to main">
      <img src="{logo}" alt="ExoMatch logo" />
      <span>ExoMatch</span>
    </button>
  </form>

  <!-- 右上三槓 -->
  <input type="checkbox" id="exo-toggle" aria-controls="exo-menu" aria-expanded="false"/>
  <label for="exo-toggle" class="exo-btn" title="More" aria-label="Open menu">
    <span class="exo-bars"></span>
  </label>

  <!-- 下拉選單（同分頁導向 + 背景框滑下來 + 圖示） -->
  <nav class="exo-menu" id="exo-menu" aria-label="More menu">
    <ul>
      <li data-path="/about">
        <form class="exo-link-form" method="get" action="/about">
          <button class="exo-link-btn" type="submit">
            <span class="exo-ico">
              <!-- info 圖示 -->
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill="currentColor" d="M12 2a10 10 0 1 0 .001 20.001A10 10 0 0 0 12 2zm0 14a1 1 0 1 1-2 0v-5a1 1 0 1 1 2 0v5zm0-8.25a1.25 1.25 0 1 1 0 2.5 1.25 1.25 0 0 1 0-2.5z"/></svg>
            </span>
            About our model
          </button>
        </form>
      </li>
      <li data-path="/analyze">
        <form class="exo-link-form" method="get" action="/analyze">
          <button class="exo-link-btn" type="submit">
            <span class="exo-ico">
              <!-- 分析/趨勢 圖示 -->
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill="currentColor" d="M3 3a1 1 0 0 1 2 0v16h16a1 1 0 1 1 0 2H5a2 2 0 0 1-2-2V3z"/><path fill="currentColor" d="M21 7.414 17.707 10.7l-3-3L8 14.41l-2.293-2.3-1.414 1.42L8 17.24l7.293-7.29 3 3L22.414 8.7 21 7.414z"/></svg>
            </span>
            Analyze your data
          </button>
        </form>
      </li>
      <li data-path="/fits_converter">
        <form class="exo-link-form" method="get" action="/fits_converter">
          <button class="exo-link-btn" type="submit">
            <span class="exo-ico">
              <!-- 望遠鏡/轉換 圖示 -->
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill="currentColor" d="M20.38 8.57l-1.23 1.85a8 8 0 0 1-.22 7.58H5.07A8 8 0 0 1 15.58 6.85l1.85-1.23A10 10 0 0 0 3.35 19a2 2 0 0 0 1.72 1h13.85a2 2 0 0 0 1.74-1 10 10 0 0 0-.27-10.44z"/><path fill="currentColor" d="M10.59 15.41a2 2 0 0 0 2.83 0l5.66-8.49-8.49 5.66a2 2 0 0 0 0 2.83z"/></svg>
            </span>
            FITS Converter
          </button>
        </form>
      </li>
      <li data-path="/vetting">
        <form class="exo-link-form" method="get" action="/vetting">
          <button class="exo-link-btn" type="submit">
            <span class="exo-ico">
              <!-- 檢核/check 圖示 -->
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill="currentColor" d="M20.285 2.859a1 1 0 0 1 .14 1.408l-10 12a1 1 0 0 1-1.43.09l-5-4.5a1 1 0 1 1 1.33-1.49l4.246 3.822 9.3-11.16a1 1 0 0 1 1.414-.17z"/></svg>
            </span>
            How model works
          </button>
        </form>
      </li>
    </ul>
  </nav>

  <!-- 點擊空白處可關閉（CSS + JS） -->
  <div class="exo-overlay" aria-hidden="true"></div>

  <!-- 頂部極細載入條（導覽點擊時顯示） -->
  <div id="exo-loader" style="
    position: absolute; left:0; right:0; top:0; height:2px;
    background: linear-gradient(90deg,#4facfe,#667eea);
    transform: scaleX(0); transform-origin: 0 50%;
    transition: transform .25s ease;
  "></div>
</div>

<script>
// 更新 aria-expanded
const toggle = document.getElementById('exo-toggle');
const menu = document.getElementById('exo-menu');
const overlay = document.querySelector('.exo-overlay');
const loader = document.getElementById('exo-loader');

function setExpanded() {{
  const expanded = toggle.checked ? 'true' : 'false';
  toggle.setAttribute('aria-expanded', expanded);
}}
toggle.addEventListener('change', setExpanded);
setExpanded();

 // 外點擊關閉 + Esc 關閉
overlay && overlay.addEventListener('click', () => {{ toggle.checked = false; setExpanded(); }});
document.addEventListener('keydown', (e) => {{
  if (e.key === 'Escape' && toggle.checked) {{
    toggle.checked = false; setExpanded();
  }}
}});

// 目前路徑高亮：比對 pathname 與 data-path 前綴
(function highlightActive() {{
  const path = window.location.pathname || '/';
  document.querySelectorAll('.exo-menu [data-path]').forEach(li => {{
    const p = li.getAttribute('data-path');
    if (p && (path === p || path.startsWith(p + '/'))) {{
      li.classList.add('is-active');
    }}
  }});
}})();

// 點擊導覽按鈕時顯示頂部極細 loading bar（不阻擋原本的表單提交）
document.querySelectorAll('.exo-link-btn, .exo-center-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    if (loader) {{
      loader.style.transform = 'scaleX(1)';
      // 1.5秒後自動隱藏（若已導頁則無感）
      setTimeout(() => {{ loader.style.transform = 'scaleX(0)'; }}, 1500);
    }}
  }});
}});
</script>
    """.format(bg=PRIMARY_BG, logo=logo), unsafe_allow_html=True)