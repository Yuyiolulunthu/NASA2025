"""
Exoplanet Vetting Platform — Enterprise (No Emoji)
Transit selector above the chart, and radio option text forced white.
Run:
    streamlit run vetting_only.py
"""

import time
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from components.banner import render_banner # Assumed available, otherwise remove or mock

# ========== Page Config ==========
st.set_page_config(
    page_title="Exoplanet Hunter — Vetting",
    page_icon="Web/logo.png", # Placeholder, ensure path is correct
    layout="wide",
    initial_sidebar_state="expanded",
)
# Assuming render_banner exists or removing it if it causes issues outside of the original environment
render_banner() 
hide_streamlit_header_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_header_style, unsafe_allow_html=True)
pio.templates.default = "plotly_dark"

# ========== CSS ==========
st.markdown(
    """
<style>
  :root{
    --bg-0:#0b1020; --bg-1:#111833;
    --panel:#121a36; --panel-2:#141c3b;
    --text-0:#e8ecf3; --text-1:#c8d0e3; --text-2:#8b95af;
    --brand:#4d7cff; --accent:#2bd4a7; --ring:#27376f; --border:#1f2c5b;
  }
  .stApp{ background: radial-gradient(1100px 900px at 10% 0%, var(--bg-1), var(--bg-0)); color:var(--text-0); }
  [data-testid="stSidebar"]{ background: linear-gradient(180deg, var(--bg-1) 0%, var(--bg-0) 100%); border-right:1px solid var(--ring); }

  /* === Radio 整段文字改為白色（含未選取項） === */
  [data-testid="stRadio"] * { color:#ffffff !important; }
  [data-testid="stRadio"] div[role="radiogroup"] label > div:nth-child(2),
  [data-testid="stRadio"] label span { color:#ffffff !important; font-weight:500; }

  /* Top bar */
  body > .main, div.block-container, main[role="main"]{ padding-bottom:76px !important; padding-top:0 !important; }
  #top-progress{ position:fixed; bottom:0; left:0; right:0; top:auto; height:68px; z-index:9999; pointer-events:none; }
  #top-progress .wrap{
    height:100%; display:flex; align-items:center; gap:16px; padding:0 20px;
    background:rgba(10,16,32,.72); backdrop-filter:blur(6px);
    border-top:1px solid var(--ring); box-shadow:0 -8px 24px rgba(0,0,0,.4);
  }
  #top-progress .label{ color:var(--text-0); font-weight:700; min-width:220px; letter-spacing:.2px; pointer-events:auto; }
  #top-progress .bar{ flex:1; height:10px; background:rgba(255,255,255,.06); border-radius:999px; overflow:hidden; }
  #top-progress .bar>i{ display:block; height:100%; background:linear-gradient(90deg, var(--brand), #7fa1ff); border-radius:999px; transition:width 380ms ease; }

  /* Cards */
  .card{ background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%); border:1px solid var(--border);
         border-radius:14px; padding:16px; box-shadow:0 6px 22px rgba(0,0,0,.30); }
  .metric{ background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%); border:1px solid var(--border);
           border-radius:12px; padding:12px; }
  .m-title{ color:var(--text-2); font-size:.85rem; margin:0 0 .2rem 0; }
  .m-val{ color:#bcd2ff; font-size:1.2rem; font-weight:700; letter-spacing:.2px; }

  /* Confidence bar */
  .conf-wrap{ margin:8px 0 2px 0; }
  .conf-bar{ height:26px; border-radius:999px; background:linear-gradient(90deg, #ef4444, #f59e0b, #10b981); position:relative; overflow:hidden; border:1px solid var(--ring);}
  .conf-ind{ position:absolute; top:0; left:0; height:100%; background:rgba(255,255,255,.25); border-right:3px solid #fff; box-shadow:0 0 16px rgba(255,255,255,.35); }
  .muted{ color:var(--text-2); }

  /* Buttons（一般） */
  .stButton>button{ border-radius:10px !important; border:1px solid var(--ring) !important; background:rgba(255,255,255,.05) !important; color:var(--text-0) !important; }

  /* === Sidebar Links 白色化 === */
  [data-testid="stSidebar"] a,
  [data-testid="stSidebar"] a span,
  [data-testid="stSidebar"] a > div,
  [data-testid="stSidebar"] button[role="button"],
  [data-testid="stSidebar"] button[role="button"] span,
  [data-testid="stSidebar"] [role="navigation"] a,
  [data-testid="stSidebar"] [role="navigation"] button {
      color: #ffffff !important;
      fill: #ffffff !important;
      opacity: 1 !important;
  }
  [data-testid="stSidebar"] button[role="button"][aria-current="true"],
  [data-testid="stSidebar"] a[aria-current="true"] {
      color: #ffffff !important;
      font-weight: 700 !important;
  }
  [data-testid="stSidebar"] svg {
      fill: #ffffff !important;
      color: #ffffff !important;
  }

/* === 強化 Submit Decision hover 動畫 (確保匹配最新 Streamlit 結構) === */
.stButton button[data-testid="baseButton-primary"],
.stButton > button[kind="primary"],
button[kind="primary"],
button[data-testid="baseButton-primary"] {
  display: inline-block;
  height: 64px !important;
  font-size: 1.2rem !important;
  font-weight: 700 !important;
  border-radius: 10px !important;
  color: #fff !important;
  background: linear-gradient(90deg, var(--brand), #7fa1ff) !important;
  border: 1px solid var(--ring) !important;
  box-shadow: 0 0 16px rgba(77,124,255,.45) !important;
  transition: all 0.25s ease !important;
  transform-origin: center center;
  will-change: transform, box-shadow, filter;
}

    /* hover 動畫（加 !important 且包含不同層級） */
    .stButton button[data-testid="baseButton-primary"]:hover,
    .stButton > button[kind="primary"]:hover,
    button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover {
    transform: scale(1.05) translateZ(0) !important;
    box-shadow: 0 0 24px rgba(77,124,255,.75), 0 6px 24px rgba(0,0,0,.35) !important;
    filter: brightness(1.05) saturate(1.08) !important;
    }

    /* active 點擊回饋 */
    .stButton button[data-testid="baseButton-primary"]:active,
    .stButton > button[kind="primary"]:active,
    button[kind="primary"]:active,
    button[data-testid="baseButton-primary"]:active {
    transform: scale(0.98) translateY(1px) !important;
    box-shadow: 0 0 18px rgba(77,124,255,.5) inset !important;
    filter: brightness(0.9) !important;
    }

    /* 統一 Submit 高度與狀態，不因 primary/secondary 變動 */
    .submit-wide .stButton > button {
    height:72px !important;
    min-height:72px !important;
    line-height:72px !important;
    width:100% !important;
    font-size:1.15rem !important;
    font-weight:800 !important;
    border-radius:14px !important;
    border:1px solid var(--ring) !important;
    box-shadow:none !important;        /* 避免陰影造成視覺高度變化 */
    transform:none !important;         /* 避免 scale 造成跳動 */
    filter:none !important;
    }

    /* Submit 的 hover/active 也不要放大 */
    .submit-wide .stButton > button:hover,
    .submit-wide .stButton > button:active,
    .submit-wide .stButton > button:focus {
    transform:none !important;
    box-shadow:0 4px 12px rgba(0,0,0,.18) !important;
    filter:brightness(1.02) !important;
    }
    
</style>
""",
    unsafe_allow_html=True,
)

# ========== Demo Data ==========
if "candidate_index" not in st.session_state:
    st.session_state.candidate_index = 0
if "candidates" not in st.session_state:
    st.session_state.candidates = []
    rng = np.random.default_rng(7)
    for i in range(10):
        t = np.linspace(0, 50, 500)
        f = 1 + rng.normal(0, 0.002, 500)
        period = rng.uniform(5, 30)
        events = []
        for tt in np.arange(0, 50, period):
            mask = (t > tt - 1) & (t < tt + 1)
            f[mask] -= rng.uniform(0.005, 0.02)
            events.append(float(tt))
        st.session_state.candidates.append(
            {
                "id": f"TIC-{200000+i}",
                "time": t,
                "flux": f,
                "period": float(period),
                "depth": float(rng.uniform(0.5, 2.0)),
                "duration": float(rng.uniform(2, 6)),
                "snr": float(rng.uniform(10, 50)),
                "radius_ratio": float(rng.uniform(0.05, 0.15)),
                "ai_confidence": float(rng.uniform(0.5, 0.95)),
                "transit_times": events[:3],
                "color_index": float(rng.uniform(0.5, 1.5)),
                "effective_temp": float(rng.uniform(4000, 7000)),
            }
        )
if "labels" not in st.session_state:
    st.session_state.labels = []

# ========== Utils ==========
def create_interactive_lightcurve(c):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=c["time"], y=c["flux"], mode="lines", name="Flux",
            line=dict(width=1.6),
            hovertemplate="Time: %{x:.2f} d<br>Flux: %{y:.5f}<extra></extra>",
        )
    )
    for i, t0 in enumerate(c["transit_times"]):
        fig.add_vrect(
            x0=t0 - 1, x1=t0 + 1, fillcolor="rgba(239,68,68,0.18)", layer="below", line_width=0,
            annotation_text=f"T{i+1}", annotation_position="top left", annotation=dict(font_size=10),
        )
    fig.update_layout(
        title=dict(text="Full Light Curve", font=dict(size=16)),
        xaxis_title="Time (days)", yaxis_title="Normalized Flux",
        height=480, hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(gridcolor="rgba(200,200,200,0.15)")
    fig.update_yaxes(gridcolor="rgba(200,200,200,0.15)")
    return fig


def create_transit_zoom(c, k=0, height=520):
    if len(c["transit_times"]) == 0:
        return go.Figure()
    if k >= len(c["transit_times"]):
        k = 0
    t_center = c["transit_times"][k]
    mask = (c["time"] > t_center - 2) & (c["time"] < t_center + 2)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=c["time"][mask], y=c["flux"][mask], mode="markers+lines",
            marker=dict(size=4), line=dict(width=2),
            hovertemplate="Time: %{x:.3f} d<br>Flux: %{y:.6f}<extra></extra>",
        )
    )
    half = (c["duration"] / 2.0) / 24.0
    fig.add_vrect(
        x0=t_center - half, x1=t_center + half,
        fillcolor="rgba(16,185,129,0.18)", layer="below", line_width=0,
        annotation_text="Estimated Duration Window", annotation_position="top",
        annotation=dict(font_size=10),
    )
    fig.update_layout(
        title=dict(text=f"Transit {k+1} — High-Resolution", font=dict(size=14)),
        xaxis_title="Time (days)", yaxis_title="Normalized Flux",
        height=height, hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
        margin=dict(l=40, r=20, t=40, b=36),
    )
    fig.update_xaxes(gridcolor="rgba(200,200,200,0.15)")
    fig.update_yaxes(gridcolor="rgba(200,200,200,0.15)")
    return fig


def confidence_bar(conf):
    return f"""
    <div class="card conf-wrap">
      <div style='display:flex;justify-content:space-between;align-items:end;margin-bottom:.6rem;'>
        <div style="margin:0; color:var(--text-1);">AI Confidence Score</div>
        <div style='color:#bcd2ff; font-size:1.05rem; font-weight:700;'>{conf:.1%}</div>
      </div>
      <div class='conf-bar'><div class='conf-ind' style='width:{conf*100:.2f}%;'></div></div>
      <div style='display:flex;justify-content:space-between;margin-top:.35rem;font-size:.8rem;' class='muted'>
        <span>Low</span><span>Medium</span><span>High</span>
      </div>
    </div>
    """


def metric_html(title, value):
    return f"<div class='metric'><div class='m-title'>{title}</div><div class='m-val'>{value}</div></div>"

# ========== Main ==========
def render_vetting():
    candidates = st.session_state.candidates
    idx = st.session_state.candidate_index
    total = len(candidates)

    # cap display values so they never exceed total / 100%
    if total:
        pct = min(int(((idx + 1) / total) * 100), 100)
        display_num = min(idx + 1, total)
    else:
        pct = 0
        display_num = 0

    st.markdown(
        f"""
        <div id="top-progress">
          <div class="wrap">
            <div class="label">Candidate {display_num} / {total} — {pct}%</div>
            <div class="bar"><i style="width:{pct}%;"></i></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown("#### Stellar Parameters")
        if idx < total:
            cur = candidates[idx]
            st.markdown(
                f"""
                <div class='card'>
                  <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <h4 style='margin:0;color:#bcd2ff;'>{cur['id']}</h4>
                    <span class='muted' style='font-size:.85rem;'>SNR {cur['snr']:.1f}</span>
                  </div>
                  <div style='height:10px;'></div>
                  <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;'>
                    {metric_html("B−V (mag)", f"{cur['color_index']:.3f}")}
                    {metric_html("T_eff (K)", f"{cur['effective_temp']:.0f}")}
                    {metric_html("SNR", f"{cur['snr']:.1f}")}
                    {metric_html("Rp/R★", f"{cur['radius_ratio']:.3f}")}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if idx >= total:
        st.markdown("<h1 style='text-align:center;margin-top:14vh;'>Vetting Complete</h1>", unsafe_allow_html=True)
        return

    cur = candidates[idx]

    st.markdown(
        f"""
        <div class='card' style='padding:12px 16px; margin-top:8px;'>
          <div style='display:flex;justify-content:space-between;align-items:center; gap:12px;'>
            <div>
              <div style='color:var(--text-1);margin:0;'>Target</div>
              <div style='font-weight:700; font-size:1.15rem; color:#bcd2ff;'>{cur['id']}</div>
            </div>
            <div style='display:flex; gap:12px;'>
              {metric_html("Period", f"{cur['period']:.2f} d")}
              {metric_html("Depth", f"{cur['depth']:.2f} %")}
              {metric_html("Duration", f"{cur['duration']:.2f} hr")}
              {metric_html("Rp/R★", f"{cur['radius_ratio']:.3f}")}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(confidence_bar(cur["ai_confidence"]), unsafe_allow_html=True)

    st.subheader("Full Light Curve")
    fig_full = create_interactive_lightcurve(cur)
    st.plotly_chart(fig_full, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})

    # === Transit Detail ===
    st.subheader("Transit Detail")
    with st.container():
        head_l, head_r = st.columns([6, 2])
        with head_l:
            st.caption("Select a transit window to inspect the high-resolution view.")
        with head_r:
            transit_num = st.selectbox(
                "Transit Window",
                range(max(1, len(cur["transit_times"]))),
                index=0,
                format_func=lambda x: f"Transit {x+1}",
                key=f"transit_select_{idx}",
                label_visibility="visible",
            )

        fig_zoom = create_transit_zoom(cur, transit_num, height=520)
        st.plotly_chart(fig_zoom, use_container_width=True, config={"displayModeBar": False})

render_vetting()

# === Decision ===
st.markdown("---")
st.subheader("Vetting Decision")

idx = st.session_state.get("candidate_index", 0)

labels = ["False Positive", "Planet Candidate", "Confirmed Planet"]
mapping = {"False Positive": "FALSE_POSITIVE",
           "Planet Candidate": "CANDIDATE",
           "Confirmed Planet": "CONFIRMED"}

sel_key = f"selected_label_{idx}"
if sel_key not in st.session_state:
    st.session_state[sel_key] = None

# --- STEP 1: 選擇列（單行） ---
st.markdown("<div style='color:var(--text-0); font-size:1.05rem; font-weight:700; margin-bottom:6px;'>STEP 1: Select your vetting decision below</div>", unsafe_allow_html=True)
# keep radio as a single horizontal row; key is per-candidate so state is preserved
chosen = st.radio(
    "",  # no extra label because we already printed STEP 1 above
    options=labels,
    index=None,
    horizontal=True,
    key=sel_key,
)

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

# --- STEP 2: Submit 列（單行） ---
st.markdown("<div style='color:var(--text-0); font-size:1.05rem; font-weight:700; margin-bottom:6px;'>STEP 2: Submit your decision</div>", unsafe_allow_html=True)
st.markdown("<div class='submit-wide'>", unsafe_allow_html=True)

# read chosen from session_state to ensure consistent behavior after rerun
chosen = st.session_state.get(sel_key)
submit_type = "primary" if chosen is not None else "secondary"

# single Submit button (unique key) — removed duplicate earlier submit button to avoid double-click issues
if st.button(
    "Submit Decision",
    type=submit_type,
    use_container_width=True,
    key=f"submit_btn_{idx}_step2",
    disabled=(chosen is None),
):
    # guard just in case
    if chosen is not None:
        st.session_state.labels.append(mapping[chosen])
        st.session_state.candidate_index += 1
        # clear the next candidate's selected key so radio starts empty for the next candidate
        next_key = f"selected_label_{st.session_state.candidate_index}"
        st.session_state.pop(next_key, None)
        st.toast(f"Recorded: {chosen}")
        time.sleep(0.1)
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

# --- Navigation Buttons（單行） ---
st.markdown("<div style='color:var(--text-0); font-size:1.05rem; font-weight:700; margin-bottom:6px;'>You can navigate between candidates!</div>", unsafe_allow_html=True)
st.markdown("<div class='action-row'>", unsafe_allow_html=True)
col_prev, col_skip, col_reset = st.columns([1, 1, 1])
with col_prev:
    if st.button("Previous", use_container_width=True, disabled=(idx == 0), key=f"prev_{idx}"):
        st.session_state.candidate_index = max(0, idx - 1)
        # remove last label if exist to keep labels in sync
        if st.session_state.labels:
            st.session_state.labels.pop()
        # remove selected key for the (now current) candidate so radio shows correct state
        prev_key = f"selected_label_{st.session_state.candidate_index}"
        st.session_state.pop(prev_key, None)
        st.rerun()
with col_skip:
    if st.button("Skip", use_container_width=True, key=f"skip_{idx}"):
        st.session_state.candidate_index += 1
        next_key = f"selected_label_{st.session_state.candidate_index}"
        st.session_state.pop(next_key, None)
        st.rerun()
with col_reset:
    if st.button("Reset Demo Data", use_container_width=True, key=f"reset_{idx}"):
        st.session_state.clear()
        st.rerun()
st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    
    """
<div style='text-align:center;color:var(--text-2);padding:18px 0;'>
  <div style='font-size:.9rem;opacity:.7;'>Exoplanet Hunter — Vetting Suite</div>
  <div style='font-size:.8rem;opacity:.5;'>Build v2.3.1 · Enterprise Edition</div>
</div>
""",
    unsafe_allow_html=True,
)
