"""
🌌 Professional Exoplanet Vetting Platform — Vetting Only
一執行就直接進入「Start Vetting」後的審核頁面（無首頁）。
執行：
    streamlit run vetting_only.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# ================= Page Configuration =================
st.set_page_config(
    page_title="Exoplanet Hunter — Vetting",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================= Dark Space Theme CSS =================
st.markdown(
    """
<style>
  .stApp { background: linear-gradient(135deg,#0a0e27 0%,#16213e 50%,#0f3460 100%); color:#e0e0e0; }
  .stApp::before { content:""; position:fixed; inset:0; background-image:
      radial-gradient(2px 2px at 20px 30px,#eee,rgba(0,0,0,0)),
      radial-gradient(2px 2px at 60px 70px,#fff,rgba(0,0,0,0)),
      radial-gradient(1px 1px at 50px 50px,#fff,rgba(0,0,0,0));
      background-repeat:repeat; background-size:200px 200px; opacity:.4; z-index:-1;
      animation: twinkle 3s ease-in-out infinite; }
  @keyframes twinkle { 0%,100%{opacity:.3;} 50%{opacity:.6;} }
  .candidate-card { background: linear-gradient(135deg, rgba(22,33,62,.95), rgba(15,52,96,.95));
      border:2px solid rgba(102,126,234,.3); border-radius:20px; padding:2rem; margin:1.2rem 0;
      box-shadow:0 8px 32px rgba(31,38,135,.37); backdrop-filter:blur(10px); }
  .metric-card { background: rgba(22,33,62,.6); border:1px solid rgba(102,126,234,.3);
      border-radius:15px; padding:1.2rem; text-align:center; box-shadow:0 4px 20px rgba(0,0,0,.3); }
  .confidence-bar { height:30px; border-radius:15px; background:linear-gradient(90deg,#ef4444,#f59e0b,#10b981); position:relative; overflow:hidden; }
  .confidence-indicator { position:absolute; top:0; left:0; height:100%; background:rgba(255,255,255,.3); border-right:3px solid #fff; box-shadow:0 0 20px rgba(255,255,255,.5); }
  [data-testid="stSidebar"] { background: linear-gradient(180deg,#0a0e27 0%,#16213e 100%); }
</style>
""",
    unsafe_allow_html=True,
)

# ================= Session State Init =================
if "candidate_index" not in st.session_state:
    st.session_state.candidate_index = 0
if "candidates" not in st.session_state:
    # Demo 候選資料；之後可改為讀檔或 API
    st.session_state.candidates = []
    rng = np.random.default_rng(7)
    for i in range(10):
        time_data = np.linspace(0, 50, 500)
        flux = 1 + rng.normal(0, 0.002, 500)
        period = rng.uniform(5, 30)
        transit_times = []
        for t in np.arange(0, 50, period):
            mask = (time_data > t - 1) & (time_data < t + 1)
            flux[mask] -= rng.uniform(0.005, 0.02)
            transit_times.append(float(t))
        st.session_state.candidates.append(
            {
                "id": f"TIC-{200000+i}",
                "time": time_data,
                "flux": flux,
                "period": float(period),
                "depth": float(rng.uniform(0.5, 2.0)),
                "duration": float(rng.uniform(2, 6)),
                "snr": float(rng.uniform(10, 50)),
                "radius_ratio": float(rng.uniform(0.05, 0.15)),
                "ai_confidence": float(rng.uniform(0.5, 0.95)),
                "transit_times": transit_times[:3],
                "color_index": float(rng.uniform(0.5, 1.5)),
                "effective_temp": float(rng.uniform(4000, 7000)),
            }
        )
if "labels" not in st.session_state:
    st.session_state.labels = []

# ================= Utils =================

def create_interactive_lightcurve(candidate):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=candidate["time"], y=candidate["flux"], mode="lines", name="Flux",
            line=dict(color="#4facfe", width=1.5),
            hovertemplate="Time: %{x:.2f} days<br>Flux: %{y:.4f}<extra></extra>",
        )
    )
    for i, t in enumerate(candidate["transit_times"]):
        fig.add_vrect(
            x0=t - 1, x1=t + 1, fillcolor="rgba(239,68,68,0.2)", layer="below", line_width=0,
            annotation_text=f"Transit {i+1}", annotation_position="top left",
            annotation=dict(font_size=10, font_color="#ef4444"),
        )
    fig.update_layout(
        title={"text": "Full Light Curve (Interactive - Zoom & Pan Enabled)", "font": {"size": 16, "color": "#667eea"}},
        xaxis_title="Time (days)", yaxis_title="Normalized Flux", height=400,
        paper_bgcolor="rgba(10,14,39,0.8)", plot_bgcolor="rgba(22,33,62,0.6)",
        font=dict(color="#e0e0e0", size=12), hovermode="x unified",
        xaxis=dict(gridcolor="rgba(102,126,234,0.2)", rangeslider=dict(visible=True, bgcolor="rgba(22,33,62,0.4)")),
        yaxis=dict(gridcolor="rgba(102,126,234,0.2)"),
    )
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)
    return fig


def create_transit_zoom(candidate, transit_index=0):
    if transit_index >= len(candidate["transit_times"]):
        transit_index = 0
    t_center = candidate["transit_times"][transit_index]
    mask = (candidate["time"] > t_center - 2) & (candidate["time"] < t_center + 2)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=candidate["time"][mask], y=candidate["flux"][mask], mode="markers+lines", name="Transit Detail",
            line=dict(color="#f093fb", width=2), marker=dict(size=4, color="#f093fb"),
            hovertemplate="Time: %{x:.3f} days<br>Flux: %{y:.5f}<extra></extra>",
        )
    )
    fig.add_vrect(
        x0=t_center - candidate["duration"] / 48, x1=t_center + candidate["duration"] / 48,
        fillcolor="rgba(16,185,129,0.15)", layer="below", line_width=0, annotation_text="Transit Duration", annotation_position="top",
    )
    fig.update_layout(
        title={"text": f"Transit {transit_index+1} - High Resolution View", "font": {"size": 14, "color": "#f093fb"}},
        xaxis_title="Time (days)", yaxis_title="Normalized Flux", height=350,
        paper_bgcolor="rgba(10,14,39,0.8)", plot_bgcolor="rgba(22,33,62,0.6)",
        font=dict(color="#e0e0e0", size=11), hovermode="x unified",
        xaxis=dict(gridcolor="rgba(102,126,234,0.2)"), yaxis=dict(gridcolor="rgba(102,126,234,0.2)"),
    )
    return fig


def confidence_bar(conf):
    return f"""
    <div style='margin: 1rem 0;'>
      <div style='display:flex;justify-content:space-between;margin-bottom:.5rem;'>
        <span style='color:#a0a0a0;font-size:.9rem;'>AI Confidence Score</span>
        <span style='color:#667eea;font-size:1.2rem;font-weight:bold;'>{conf:.1%}</span>
      </div>
      <div class='confidence-bar'><div class='confidence-indicator' style='width:{conf*100}%;'></div></div>
      <div style='display:flex;justify-content:space-between;margin-top:.3rem;font-size:.8rem;color:#666;'>
        <span>Low</span><span>Medium</span><span>High</span>
      </div>
    </div>
    """

# ================= Vetting Page (Only) =================

def render_vetting():
    candidates = st.session_state.candidates
    idx = st.session_state.candidate_index

    # Fixed top progress bar (will stay at top when scrolling)
    total = len(candidates) if candidates else 0
    top_progress = (idx + 1) / total if total > 0 else 0
    pct = int(top_progress * 100)
    st.markdown(f"""
    <style>
      /* reserve space so app content isn't hidden under the fixed bar */
      body > .main, div.block-container, main[role="main"] {{ padding-top: 72px !important; }}
      #top-progress {{ position: fixed; top: 0; left: 0; right: 0; height: 64px; z-index: 9999; pointer-events: none; }}
      #top-progress .wrap {{ height: 100%; display: flex; align-items: center; gap: 12px; padding: 0 18px;
                           background: rgba(10,14,39,0.70); backdrop-filter: blur(6px); box-shadow: 0 6px 20px rgba(2,6,23,0.6); }}
      #top-progress .label {{ color: #e6f7ff; font-weight: 700; pointer-events: auto; min-width:160px; white-space:nowrap; }}
      #top-progress .bar {{ flex: 1; height: 10px; background: rgba(255,255,255,0.06); border-radius: 999px; overflow: hidden; }}
      #top-progress .bar > i {{ display: block; height: 100%; width: {pct}%; background: linear-gradient(90deg,#4facfe,#667eea); border-radius: 999px; transition: width 400ms ease; }}
    </style>
    <div id="top-progress">
      <div class="wrap">
        <div class="label">Candidate {idx+1} / {total} — {pct}%</div>
        <div class="bar"><i></i></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar — Stellar parameters & stats
    with st.sidebar:
        st.markdown("### 🌟 Stellar Parameters")
        if idx < len(candidates):
            cur = candidates[idx]
            st.markdown(
                f"""
                <div class='metric-card'>
                  <h4 style='color:#667eea;margin-bottom:1rem;'>{cur['id']}</h4>
                  <div style='text-align:left;padding:.5rem;'>
                    <p><strong>Color Index (B−V):</strong><br>{cur['color_index']:.3f}</p>
                    <p><strong>Effective Temp:</strong><br>{cur['effective_temp']:.0f} K</p>
                    <p><strong>SNR:</strong><br>{cur['snr']:.1f}</p>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("---")
            st.markdown("### 📊 Quick Stats")
            st.metric("Current Index", f"{idx+1} / {len(candidates)}")
            st.metric("Your Labels", len(st.session_state.labels))
        st.markdown("---")
        st.markdown("### ℹ️ Instructions")
        st.info(
            """
**Zoom & Pan**
- Scroll to zoom
- Click & drag to pan
- Double-click to reset

**Vetting**
- Review light curve
- Check transit depth
- Examine periodicity
- Make judgment
            """,
        )

    # Main content
    if idx >= len(candidates):
        st.markdown("<h1 style='text-align:center;'>🎉 VETTING COMPLETE!</h1>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='candidate-card' style='text-align:center;'>
              <h2 style='color:#667eea;'>All candidates reviewed!</h2>
              <p style='font-size:1.2rem;margin:1.2rem 0;'>Thank you for your contribution! You've helped make the AI smarter 🚀</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    cur = candidates[idx]

    # Progress
    progress = (idx + 1) / len(candidates)
    st.progress(progress)
    st.markdown(
        f"""
        <div style='text-align:center;margin:1rem 0;'>
          <span style='color:#667eea;font-size:1.2rem;font-weight:bold;'>CANDIDATE {idx+1} / {len(candidates)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # AI Confidence
    st.markdown(confidence_bar(cur["ai_confidence"]), unsafe_allow_html=True)

    # Light curve — full
    st.markdown('<h3 style="margin:0;">📈 Interactive Light Curve Analysis</h3>', unsafe_allow_html=True)
    fig_full = create_interactive_lightcurve(cur)
    st.plotly_chart(fig_full, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})

    # Transit detail
    st.markdown('<h3 style="margin:0;">🔍 Transit Detail View</h3>', unsafe_allow_html=True)
    col_a, col_b = st.columns([3, 1])
    with col_b:
        transit_num = st.selectbox("Select Transit", range(len(cur["transit_times"])), format_func=lambda x: f"Transit {x+1}")
    fig_zoom = create_transit_zoom(cur, transit_num)
    st.plotly_chart(fig_zoom, use_container_width=True, config={"displayModeBar": False})

    # Physical parameters
    st.markdown('<h3 style="margin:0;">🔬 Physical Parameters</h3>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    params = [
        ("Orbital Period", f"{cur['period']:.2f} days", c1),
        ("Transit Depth", f"{cur['depth']:.2f}%", c2),
        ("Duration", f"{cur['duration']:.2f} hrs", c3),
        ("Radius Ratio (Rp/R*)", f"{cur['radius_ratio']:.3f}", c4),
    ]
    for label, value, col in params:
        with col:
            st.markdown(
                f"""
                <div class='metric-card'>
                  <div style='color:#a0a0a0;font-size:.85rem;'>{label}</div>
                  <div style='color:#667eea;font-size:1.3rem;font-weight:bold;margin-top:.5rem;'>{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Decisions
    st.markdown('<h3 style="margin:0;">🎯 Your Vetting Decision</h3>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1:
        if st.button("👈 FALSE POSITIVE", use_container_width=True, key="fp"):
            st.session_state.labels.append("FALSE_POSITIVE")
            st.session_state.candidate_index += 1
            st.success("✅ Marked as False Positive")
            time.sleep(0.2)
            st.rerun()
    with d2:
        if st.button("👉 PLANET CANDIDATE", use_container_width=True, key="candidate"):
            st.session_state.labels.append("CANDIDATE")
            st.session_state.candidate_index += 1
            st.info("✅ Marked as Candidate")
            time.sleep(0.2)
            st.rerun()
    with d3:
        if st.button("👆 CONFIRMED PLANET", use_container_width=True, key="confirmed", type="primary"):
            st.session_state.labels.append("CONFIRMED")
            st.session_state.candidate_index += 1
            st.balloons()
            st.success("✅ Confirmed as Planet!")
            time.sleep(0.2)
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Navigation
    n1, n2, n3 = st.columns(3)
    with n1:
        if st.button("⬅️ Previous", use_container_width=True, disabled=(idx == 0)):
            st.session_state.candidate_index -= 1
            if st.session_state.labels:
                st.session_state.labels.pop()
            st.rerun()
    with n2:
        if st.button("⏭️ Skip", use_container_width=True):
            st.session_state.candidate_index += 1
            st.rerun()
    with n3:
        if st.button("🧪 Reset Demo Data", use_container_width=True):
            del st.session_state["candidates"]
            del st.session_state["labels"]
            st.session_state.candidate_index = 0
            st.rerun()

# Render
render_vetting()

# Footer
st.markdown(
    """
<div style='text-align:center;color:#667eea;padding:2rem;'>
  <p style='font-size:.9rem;opacity:.6;'>🌌 EXOPLANET HUNTER v2.1 — Vetting Only</p>
</div>
""",
    unsafe_allow_html=True,
)
