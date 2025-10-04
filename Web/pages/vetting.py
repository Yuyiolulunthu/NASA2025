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
from components.banner import render_banner

# ========== Page Config ==========
st.set_page_config(
    page_title="Exoplanet Hunter — Vetting",
    page_icon="Web/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)
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
  /* 更精準：僅作用在文字，不動圓點本身樣式 */
  [data-testid="stRadio"] div[role="radiogroup"] label > div:nth-child(2),
  [data-testid="stRadio"] label span { color:#ffffff !important; font-weight:500; }

  /* Top bar */
  body > .main, div.block-container, main[role="main"]{ padding-top:76px !important; }
  #top-progress{ position:fixed; top:0; left:0; right:0; height:68px; z-index:9999; pointer-events:none; }
  #top-progress .wrap{
    height:100%; display:flex; align-items:center; gap:16px; padding:0 20px;
    background:rgba(10,16,32,.72); backdrop-filter:blur(6px);
    border-bottom:1px solid var(--ring); box-shadow:0 8px 24px rgba(0,0,0,.4);
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

  /* Buttons */
  .stButton>button{ border-radius:10px !important; border:1px solid var(--ring) !important; background:rgba(255,255,255,.05) !important; color:var(--text-0) !important; }
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
    pct = int(((idx + 1) / total) * 100) if total else 0

    st.markdown(
        f"""
        <div id="top-progress">
          <div class="wrap">
            <div class="label">Candidate {idx+1} / {total} — {pct}%</div>
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
    st.progress((idx + 1) / total, text=f"Candidate {idx+1} of {total}")

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

    # === Decision ===
    st.subheader("Vetting Decision")
    decision = st.radio(
        "Select a label",
        options=["False Positive", "Planet Candidate", "Confirmed Planet"],
        horizontal=True,
        key=f"decision_{idx}",
    )

    col_submit, col_prev, col_skip, col_reset = st.columns([2, 1, 1, 1])
    with col_submit:
        if st.button("Submit Decision", use_container_width=True):
            mapping = {
                "False Positive": "FALSE_POSITIVE",
                "Planet Candidate": "CANDIDATE",
                "Confirmed Planet": "CONFIRMED",
            }
            st.session_state.labels.append(mapping[decision])
            st.session_state.candidate_index += 1
            st.toast(f"Recorded: {decision}")
            time.sleep(0.1)
            st.rerun()
    with col_prev:
        if st.button("Previous", use_container_width=True, disabled=(idx == 0)):
            st.session_state.candidate_index = max(0, idx - 1)
            if st.session_state.labels:
                st.session_state.labels.pop()
            st.rerun()
    with col_skip:
        if st.button("Skip", use_container_width=True):
            st.session_state.candidate_index += 1
            st.rerun()
    with col_reset:
        if st.button("Reset Demo Data", use_container_width=True):
            st.session_state.clear()
            st.rerun()

# ========== Run ==========
render_vetting()

st.markdown(
    """
<div style='text-align:center;color:var(--text-2);padding:18px 0;'>
  <div style='font-size:.9rem;opacity:.7;'>Exoplanet Hunter — Vetting Suite</div>
  <div style='font-size:.8rem;opacity:.5;'>Build v2.3.1 · Enterprise Edition</div>
</div>
""",
    unsafe_allow_html=True,
)
