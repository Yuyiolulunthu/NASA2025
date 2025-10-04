import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

st.set_page_config(layout="wide")

# --- SHARED FUNCTIONS, STYLES, AND HEADER ---
# NOTE: In a real-world scenario, you might put these into a shared utility file
# But for simplicity here, we'll redefine them.
def render_header():
    """Renders the fixed navigation header."""
    st.markdown("""
        <style>
            .st-emotion-cache-18ni7ap { padding-top: 80px; }
            .st-emotion-cache-16txtl3 { padding-top: 2rem; }
        </style>
    """, unsafe_allow_html=True)
    header_html = """
    <style>
        .header {
            position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
            background-color: rgba(10, 14, 39, 0.85); backdrop-filter: blur(8px);
            border-bottom: 1px solid rgba(51, 65, 85, 0.5); padding: 0.5rem 1.5rem;
            font-family: 'Inter', sans-serif;
        }
        .nav-container { display: flex; justify-content: space-between; align-items: center; max-width: 1280px; margin: auto; }
        .logo a { display: flex; align-items: center; text-decoration: none; color: white; }
        .logo span { font-weight: bold; font-size: 1.25rem; margin-left: 0.75rem; }
        .nav-links { display: flex; align-items: center; gap: 1.5rem; }
        .nav-links a { color: #D1D5DB; text-decoration: none; transition: color 0.3s; font-size: 0.9rem; }
        .nav-links a:hover { color: #818CF8; }
        .info-group { display: flex; gap: 1rem; border-left: 1px solid #4A5568; padding-left: 1.5rem; }
        .website-btn { background-color: #4F46E5; color: white; font-weight: bold; padding: 0.5rem 1.25rem; border-radius: 0.375rem; font-size: 0.875rem; transition: background-color 0.3s; }
        .website-btn:hover { background-color: #6366F1; }
        @media (max-width: 768px) { .nav-links { display: none; } }
    </style>
    <header class="header">
        <nav class="nav-container">
            <div class="logo">
                <a href="index.html" target="_blank">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#818CF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M4.93 4.93l14.14 14.14M2 12h20M4.93 19.07l14.14-14.14"></path></svg>
                    <span>InterTzailer</span>
                </a>
            </div>
            <div class="nav-links">
                <a href="about.html" target="_blank">About Us</a>
                <div class="info-group">
                    <a href="features.html" target="_blank">Features</a>
                    <a href="machine-learning.html" target="_blank">Machine Learning</a>
                    <a href="references.html" target="_blank">References</a>
                </div>
                 <a href="index.html" target="_blank" class="website-btn">Return to Website</a>
            </div>
        </nav>
    </header>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def apply_theme():
    """Applies the custom dark space theme CSS."""
    st.markdown("""
    <style>
        .candidate-card, .metric-card {
            background: linear-gradient(135deg, rgba(22, 33, 62, 0.95), rgba(15, 52, 96, 0.95));
            border: 2px solid rgba(102, 126, 234, 0.3); border-radius: 20px;
            padding: 2rem; margin: 1.5rem 0;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37); backdrop-filter: blur(10px);
        }
        .metric-card { padding: 1.5rem; text-align: center; }
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none;
            border-radius: 15px; padding: 1rem 2rem; font-size: 1.1rem; font-weight: 600;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); transition: all 0.3s ease;
        }
        .stButton > button:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6); }
        div[data-testid="column"]:nth-child(1) .stButton > button { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); }
        div[data-testid="column"]:nth-child(3) .stButton > button { background: linear-gradient(135deg, #10b981 0%, #059669 100%); }
        .confidence-bar { height: 30px; border-radius: 15px; background: linear-gradient(to right, #ef4444 0%, #f59e0b 50%, #10b981 100%); position: relative; overflow: hidden; }
        .confidence-indicator { position: absolute; top: 0; left: 0; height: 100%; background: rgba(255, 255, 255, 0.3); border-right: 3px solid white; box-shadow: 0 0 20px rgba(255, 255, 255, 0.5); }
        .space-title { font-size: 4rem; font-weight: 900; text-align: center; background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #4facfe); background-size: 300% 300%; -webkit-background-clip: text; -webkit-text-fill-color: transparent; animation: gradient-shift 3s ease infinite; padding: 2rem 0; letter-spacing: 0.1em; }
    </style>""", unsafe_allow_html=True)

def create_interactive_lightcurve(candidate):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=candidate['time'], y=candidate['flux'], mode='lines', name='Flux', line=dict(color='#4facfe', width=1.5), hovertemplate='Time: %{x:.2f} days<br>Flux: %{y:.4f}<extra></extra>'))
    for i, t in enumerate(candidate['transit_times']):
        fig.add_vrect(x0=t-1, x1=t+1, fillcolor="rgba(239, 68, 68, 0.2)", layer="below", line_width=0, annotation_text=f"Transit {i+1}", annotation_position="top left", annotation=dict(font_size=10, font_color="#ef4444"))
    fig.update_layout(title={'text': 'Full Light Curve (Interactive - Zoom & Pan Enabled)','font': {'size': 16, 'color': '#667eea'}}, xaxis_title='Time (days)', yaxis_title='Normalized Flux', height=400, paper_bgcolor='rgba(10, 14, 39, 0.8)', plot_bgcolor='rgba(22, 33, 62, 0.6)', font=dict(color='#e0e0e0', size=12), hovermode='x unified', dragmode='pan', xaxis=dict(gridcolor='rgba(102, 126, 234, 0.2)', rangeslider=dict(visible=True, bgcolor='rgba(22, 33, 62, 0.4)')), yaxis=dict(gridcolor='rgba(102, 126, 234, 0.2)'))
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)
    return fig

def create_transit_zoom(candidate, transit_index=0):
    if transit_index >= len(candidate['transit_times']): transit_index = 0
    t_center = candidate['transit_times'][transit_index]
    mask = (candidate['time'] > t_center-2) & (candidate['time'] < t_center+2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=candidate['time'][mask], y=candidate['flux'][mask], mode='markers+lines', name='Transit Detail', line=dict(color='#f093fb', width=2), marker=dict(size=4, color='#f093fb'), hovertemplate='Time: %{x:.3f} days<br>Flux: %{y:.5f}<extra></extra>'))
    fig.add_vrect(x0=t_center-candidate['duration']/48, x1=t_center+candidate['duration']/48, fillcolor="rgba(16, 185, 129, 0.15)", layer="below", line_width=0, annotation_text="Transit Duration", annotation_position="top")
    fig.update_layout(title={'text': f'Transit {transit_index+1} - High Resolution View', 'font': {'size': 14, 'color': '#f093fb'}}, xaxis_title='Time (days)', yaxis_title='Normalized Flux', height=350, paper_bgcolor='rgba(10, 14, 39, 0.8)', plot_bgcolor='rgba(22, 33, 62, 0.6)', font=dict(color='#e0e0e0', size=11), hovermode='x unified', xaxis=dict(gridcolor='rgba(102, 126, 234, 0.2)'), yaxis=dict(gridcolor='rgba(102, 126, 234, 0.2)'))
    return fig

def create_confidence_bar(confidence):
    return f"""<div style='margin: 1rem 0;'><div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'><span style='color: #a0a0a0; font-size: 0.9rem;'>AI Confidence Score</span><span style='color: #667eea; font-size: 1.2rem; font-weight: bold;'>{confidence:.1%}</span></div><div class='confidence-bar'><div class='confidence-indicator' style='width: {confidence*100}%;'></div></div><div style='display: flex; justify-content: space-between; margin-top: 0.3rem; font-size: 0.8rem; color: #666;'><span>Low</span><span>Medium</span><span>High</span></div></div>"""

# --- APP LAYOUT ---
apply_theme()
render_header()

# Check if candidates exist
if 'candidates' not in st.session_state or not st.session_state.candidates:
    st.warning("No candidates loaded. Please return to the Home page to load data.")
    st.stop()
    
candidates = st.session_state.candidates
idx = st.session_state.candidate_index

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🌟 Stellar Parameters")
    if idx < len(candidates):
        current = candidates[idx]
        st.markdown(f"""<div class="metric-card"><h4 style='color: #667eea; margin-bottom: 1rem;'>{current['id']}</h4><div style='text-align: left; padding: 0.5rem;'><p><strong>Color Index (B-V):</strong><br>{current['color_index']:.3f}</p><p><strong>Effective Temp:</strong><br>{current['effective_temp']:.0f} K</p><p><strong>SNR:</strong><br>{current['snr']:.1f}</p></div></div>""", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        st.metric("Current Index", f"{idx + 1} / {len(candidates)}")
        st.metric("Your Labels", len(st.session_state.labels))
    st.markdown("---")
    st.markdown("### ℹ️ Instructions")
    st.info("Zoom & Pan: Scroll to zoom, click & drag to pan, double-click to reset. \nVetting: Review charts, check parameters, and make your judgment.")

# --- MAIN CONTENT ---
if idx >= len(candidates):
    st.markdown('<h1 class="space-title">🎉 VETTING COMPLETE!</h1>', unsafe_allow_html=True)
    st.markdown("<div class='candidate-card' style='text-align: center;'><h2 style='color: #667eea;'>All candidates reviewed!</h2><p style='font-size: 1.3rem; margin: 2rem 0;'>Thank you for your contribution!<br>You've helped make the AI smarter 🚀</p></div>", unsafe_allow_html=True)
    if st.button("Go to Home Page"):
        st.switch_page("streamlit_app.py")
else:
    current = candidates[idx]
    progress = (idx + 1) / len(candidates)
    st.progress(progress)
    st.markdown(f"<div style='text-align: center; margin: 1rem 0;'><span style='color: #667eea; font-size: 1.2rem; font-weight: bold;'>CANDIDATE {idx + 1} / {len(candidates)}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='candidate-card'><div style='display: flex; justify-content: space-between; align-items: center;'><h2 style='color: #667eea; margin: 0;'>🪐 {current['id']}</h2><div style='text-align: right;'><div style='font-size: 0.9rem; color: #a0a0a0;'>AI Confidence</div><div style='font-size: 2rem; color: #f093fb; font-weight: bold;'>{current['ai_confidence']:.0%}</div></div></div></div>", unsafe_allow_html=True)
    st.markdown(create_confidence_bar(current['ai_confidence']), unsafe_allow_html=True)
    st.markdown("### 📈 Interactive Light Curve Analysis")
    st.plotly_chart(create_interactive_lightcurve(current), use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
    
    st.markdown("### 🔍 Transit Detail View")
    col1, col2 = st.columns([3, 1])
    with col2:
        transit_num = st.selectbox("Select Transit", range(len(current['transit_times'])), format_func=lambda x: f"Transit {x+1}")
    st.plotly_chart(create_transit_zoom(current, transit_num), use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("### 🔬 Physical Parameters")
    cols = st.columns(4)
    params = [("Orbital Period", f"{current['period']:.2f} days"),("Transit Depth", f"{current['depth']:.2f}%"),("Duration", f"{current['duration']:.2f} hrs"),("Radius Ratio (Rp/R*)", f"{current['radius_ratio']:.3f}")]
    for i, (label, value) in enumerate(params):
        with cols[i]:
            st.markdown(f"<div class='metric-card'><div style='color: #a0a0a0; font-size: 0.85rem;'>{label}</div><div style='color: #667eea; font-size: 1.3rem; font-weight: bold; margin-top: 0.5rem;'>{value}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 Your Vetting Decision")
    c1, c2, c3 = st.columns(3)
    if c1.button("👈 FALSE POSITIVE", use_container_width=True):
        st.session_state.labels.append('FALSE_POSITIVE'); st.session_state.candidate_index += 1; st.success("Marked as False Positive"); time.sleep(0.3); st.rerun()
    if c2.button("👉 PLANET CANDIDATE", use_container_width=True):
        st.session_state.labels.append('CANDIDATE'); st.session_state.candidate_index += 1; st.info("Marked as Candidate"); time.sleep(0.3); st.rerun()
    if c3.button("👆 CONFIRMED PLANET", use_container_width=True, type="primary"):
        st.session_state.labels.append('CONFIRMED'); st.session_state.candidate_index += 1; st.balloons(); st.success("Confirmed as Planet!"); time.sleep(0.3); st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    n1, n2, n3 = st.columns(3)
    if n1.button("⬅️ Previous", use_container_width=True, disabled=(idx == 0)):
        st.session_state.candidate_index -= 1; st.session_state.labels.pop(); st.rerun()
    if n2.button("⏭️ Skip", use_container_width=True):
        st.session_state.candidate_index += 1; st.rerun()
    if n3.button("🏠 Home", use_container_width=True):
        st.switch_page("streamlit_app.py")

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: #667eea; padding: 2rem;'><p style='font-size: 0.9rem; opacity: 0.6;'>🌌 EXOPLANET HUNTER v2.1 | Professional Vetting Platform</p></div>", unsafe_allow_html=True)
