"""
🌌 Professional Exoplanet Vetting Platform
AI × Human Collaboration System
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Page Configuration
st.set_page_config(
    page_title="Exoplanet Hunter",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Space Theme CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #16213e 50%, #0f3460 100%);
        color: #e0e0e0;
    }
    
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, #eee, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 60px 70px, #fff, rgba(0,0,0,0)),
            radial-gradient(1px 1px at 50px 50px, #fff, rgba(0,0,0,0));
        background-repeat: repeat;
        background-size: 200px 200px;
        opacity: 0.4;
        z-index: -1;
        animation: twinkle 3s ease-in-out infinite;
    }
    
    @keyframes twinkle {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 0.6; }
    }
    
    .space-title {
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 3s ease infinite;
        padding: 2rem 0;
        letter-spacing: 0.1em;
    }
    
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .candidate-card {
        background: linear-gradient(135deg, rgba(22, 33, 62, 0.95), rgba(15, 52, 96, 0.95));
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(10px);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    div[data-testid="column"]:nth-child(1) .stButton > button {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }
    
    div[data-testid="column"]:nth-child(3) .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    
    .metric-card {
        background: rgba(22, 33, 62, 0.6);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .confidence-bar {
        height: 30px;
        border-radius: 15px;
        background: linear-gradient(to right, #ef4444 0%, #f59e0b 50%, #10b981 100%);
        position: relative;
        overflow: hidden;
    }
    
    .confidence-indicator {
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        background: rgba(255, 255, 255, 0.3);
        border-right: 3px solid white;
        box-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e27 0%, #16213e 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'candidate_index' not in st.session_state:
    st.session_state.candidate_index = 0
if 'candidates' not in st.session_state:
    # Generate test candidates
    st.session_state.candidates = []
    for i in range(10):
        time_data = np.linspace(0, 50, 500)
        flux = 1 + np.random.normal(0, 0.002, 500)
        period = np.random.uniform(5, 30)
        transit_times = []
        for t in np.arange(0, 50, period):
            mask = (time_data > t-1) & (time_data < t+1)
            flux[mask] -= np.random.uniform(0.005, 0.02)
            transit_times.append(t)
        
        st.session_state.candidates.append({
            'id': f'TIC-{200000+i}',
            'time': time_data,
            'flux': flux,
            'period': period,
            'depth': np.random.uniform(0.5, 2.0),
            'duration': np.random.uniform(2, 6),
            'snr': np.random.uniform(10, 50),
            'radius_ratio': np.random.uniform(0.05, 0.15),
            'ai_confidence': np.random.uniform(0.5, 0.95),
            'transit_times': transit_times[:3],  # First 3 transits
            'color_index': np.random.uniform(0.5, 1.5),
            'effective_temp': np.random.uniform(4000, 7000)
        })
if 'labels' not in st.session_state:
    st.session_state.labels = []
if 'selected_transit' not in st.session_state:
    st.session_state.selected_transit = None

# Utility Functions
def create_interactive_lightcurve(candidate):
    """Create interactive zoomable light curve"""
    fig = go.Figure()
    
    # Full light curve
    fig.add_trace(go.Scatter(
        x=candidate['time'],
        y=candidate['flux'],
        mode='lines',
        name='Flux',
        line=dict(color='#4facfe', width=1.5),
        hovertemplate='Time: %{x:.2f} days<br>Flux: %{y:.4f}<extra></extra>'
    ))
    
    # Highlight transit regions with annotations
    for i, t in enumerate(candidate['transit_times']):
        # Transit box
        fig.add_vrect(
            x0=t-1, x1=t+1,
            fillcolor="rgba(239, 68, 68, 0.2)",
            layer="below",
            line_width=0,
            annotation_text=f"Transit {i+1}",
            annotation_position="top left",
            annotation=dict(font_size=10, font_color="#ef4444")
        )
    
    fig.update_layout(
        title={
            'text': 'Full Light Curve (Interactive - Zoom & Pan Enabled)',
            'font': {'size': 16, 'color': '#667eea'}
        },
        xaxis_title='Time (days)',
        yaxis_title='Normalized Flux',
        height=400,
        paper_bgcolor='rgba(10, 14, 39, 0.8)',
        plot_bgcolor='rgba(22, 33, 62, 0.6)',
        font=dict(color='#e0e0e0', size=12),
        hovermode='x unified',
        dragmode='pan',  # Enable panning
        xaxis=dict(
            gridcolor='rgba(102, 126, 234, 0.2)',
            rangeslider=dict(visible=True, bgcolor='rgba(22, 33, 62, 0.4)')
        ),
        yaxis=dict(gridcolor='rgba(102, 126, 234, 0.2)')
    )
    
    # Enable zoom and pan
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)
    
    return fig

def create_transit_zoom(candidate, transit_index=0):
    """Create zoomed view of specific transit"""
    if transit_index >= len(candidate['transit_times']):
        transit_index = 0
    
    t_center = candidate['transit_times'][transit_index]
    mask = (candidate['time'] > t_center-2) & (candidate['time'] < t_center+2)
    
    fig = go.Figure()
    
    # Transit detail
    fig.add_trace(go.Scatter(
        x=candidate['time'][mask],
        y=candidate['flux'][mask],
        mode='markers+lines',
        name='Transit Detail',
        line=dict(color='#f093fb', width=2),
        marker=dict(size=4, color='#f093fb'),
        hovertemplate='Time: %{x:.3f} days<br>Flux: %{y:.5f}<extra></extra>'
    ))
    
    # Highlight ingress and egress
    fig.add_vrect(
        x0=t_center-candidate['duration']/48, 
        x1=t_center+candidate['duration']/48,
        fillcolor="rgba(16, 185, 129, 0.15)",
        layer="below",
        line_width=0,
        annotation_text="Transit Duration",
        annotation_position="top"
    )
    
    fig.update_layout(
        title={
            'text': f'Transit {transit_index+1} - High Resolution View',
            'font': {'size': 14, 'color': '#f093fb'}
        },
        xaxis_title='Time (days)',
        yaxis_title='Normalized Flux',
        height=350,
        paper_bgcolor='rgba(10, 14, 39, 0.8)',
        plot_bgcolor='rgba(22, 33, 62, 0.6)',
        font=dict(color='#e0e0e0', size=11),
        hovermode='x unified',
        xaxis=dict(gridcolor='rgba(102, 126, 234, 0.2)'),
        yaxis=dict(gridcolor='rgba(102, 126, 234, 0.2)')
    )
    
    return fig

def create_confidence_bar(confidence):
    """Create confidence score bar"""
    html = f"""
    <div style='margin: 1rem 0;'>
        <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
            <span style='color: #a0a0a0; font-size: 0.9rem;'>AI Confidence Score</span>
            <span style='color: #667eea; font-size: 1.2rem; font-weight: bold;'>{confidence:.1%}</span>
        </div>
        <div class='confidence-bar'>
            <div class='confidence-indicator' style='width: {confidence*100}%;'></div>
        </div>
        <div style='display: flex; justify-content: space-between; margin-top: 0.3rem; font-size: 0.8rem; color: #666;'>
            <span>Low</span>
            <span>Medium</span>
            <span>High</span>
        </div>
    </div>
    """
    return html

# ==================== HOME PAGE ====================
if st.session_state.page == 'home':
    
    st.markdown('<h1 class="space-title">🌌 EXOPLANET HUNTER</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.5rem; color: #a0a0a0; letter-spacing: 0.2em;">AI × HUMAN COLLABORATION PLATFORM</p>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="candidate-card">
        <h2 style='text-align: center; color: #667eea;'>✨ OUR MISSION ✨</h2>
        <p style='text-align: center; font-size: 1.3rem; line-height: 1.8; margin-top: 1rem;'>
            Combining <strong>AI's Speed</strong> with <strong>Human Intuition</strong><br>
            To discover new worlds in our universe
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style='font-size: 3rem; margin-bottom: 0.5rem;'>🤖</div>
            <h3 style='color: #667eea;'>AI Screening</h3>
            <p style='color: #a0a0a0; margin-top: 0.5rem;'>
                Rapid analysis<br>
                Signal detection<br>
                Uncertainty flagging
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style='font-size: 3rem; margin-bottom: 0.5rem;'>👁️</div>
            <h3 style='color: #4facfe;'>Human Vetting</h3>
            <p style='color: #a0a0a0; margin-top: 0.5rem;'>
                Expert review<br>
                Intuitive swipes<br>
                Critical judgment
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style='font-size: 3rem; margin-bottom: 0.5rem;'>🔄</div>
            <h3 style='color: #f093fb;'>Collaborative Learning</h3>
            <p style='color: #a0a0a0; margin-top: 0.5rem;'>
                Model improvement<br>
                Contribution tracking<br>
                Continuous evolution
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 START VETTING", use_container_width=True, type="primary"):
            st.session_state.page = 'review'
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style='text-align: center;'>
            <div style='font-size: 2.5rem; color: #667eea; font-weight: bold;'>{len(st.session_state.labels)}</div>
            <div style='color: #a0a0a0;'>Vetted</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='text-align: center;'>
            <div style='font-size: 2.5rem; color: #4facfe; font-weight: bold;'>{len(st.session_state.candidates) - st.session_state.candidate_index}</div>
            <div style='color: #a0a0a0;'>Remaining</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        confirmed = len([l for l in st.session_state.labels if l == 'CONFIRMED'])
        st.markdown(f"""
        <div style='text-align: center;'>
            <div style='font-size: 2.5rem; color: #10b981; font-weight: bold;'>{confirmed}</div>
            <div style='color: #a0a0a0;'>Confirmed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        accuracy = np.random.uniform(80, 95) if len(st.session_state.labels) > 0 else 0
        st.markdown(f"""
        <div style='text-align: center;'>
            <div style='font-size: 2.5rem; color: #f093fb; font-weight: bold;'>{accuracy:.1f}%</div>
            <div style='color: #a0a0a0;'>Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

# ==================== VETTING PAGE ====================
elif st.session_state.page == 'review':
    
    candidates = st.session_state.candidates
    idx = st.session_state.candidate_index
    
    # Sidebar - Stellar Parameters
    with st.sidebar:
        st.markdown("### 🌟 Stellar Parameters")
        
        if idx < len(candidates):
            current = candidates[idx]
            
            st.markdown(f"""
            <div class="metric-card">
                <h4 style='color: #667eea; margin-bottom: 1rem;'>{current['id']}</h4>
                <div style='text-align: left; padding: 0.5rem;'>
                    <p><strong>Color Index (B-V):</strong><br>{current['color_index']:.3f}</p>
                    <p><strong>Effective Temp:</strong><br>{current['effective_temp']:.0f} K</p>
                    <p><strong>SNR:</strong><br>{current['snr']:.1f}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 📊 Quick Stats")
            st.metric("Current Index", f"{idx + 1} / {len(candidates)}")
            st.metric("Your Labels", len(st.session_state.labels))
        
        st.markdown("---")
        st.markdown("### ℹ️ Instructions")
        st.info("""
        **Zoom & Pan:**
        - Scroll to zoom
        - Click & drag to pan
        - Double-click to reset
        
        **Vetting:**
        - Review light curve
        - Check transit depth
        - Examine periodicity
        - Make judgment
        """)
    
    # Main Content
    if idx >= len(candidates):
        st.markdown('<h1 class="space-title">🎉 VETTING COMPLETE!</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div class="candidate-card" style="text-align: center;">
            <h2 style='color: #667eea;'>All candidates reviewed!</h2>
            <p style='font-size: 1.3rem; margin: 2rem 0;'>
                Thank you for your contribution!<br>
                You've helped make the AI smarter 🚀
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🏠 Return Home", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
    
    else:
        current = candidates[idx]
        
        # Progress
        progress = (idx + 1) / len(candidates)
        st.progress(progress)
        st.markdown(f"""
        <div style='text-align: center; margin: 1rem 0;'>
            <span style='color: #667eea; font-size: 1.2rem; font-weight: bold;'>
                CANDIDATE {idx + 1} / {len(candidates)}
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # # Candidate Header
        # st.markdown(f"""
        # <div class="candidate-card">
        #     <div style='display: flex; justify-content: space-between; align-items: center;'>
        #         <h2 style='color: #667eea; margin: 0;'>🪐 {current['id']}</h2>
        #         <div style='text-align: right;'>
        #             <div style='font-size: 0.9rem; color: #a0a0a0;'>AI Confidence</div>
        #             <div style='font-size: 2rem; color: #f093fb; font-weight: bold;'>{current['ai_confidence']:.0%}</div>
        #         </div>
        #     </div>
        # </div>
        # """, unsafe_allow_html=True)
        
        # AI Confidence Bar
        st.markdown(create_confidence_bar(current['ai_confidence']), unsafe_allow_html=True)
        
        # Light Curve Section
        # st.markdown("### 📈 Interactive Light Curve Analysis")
        st.markdown('<h3 style="color: #FFFFFF; margin: 0;">📈 Interactive Light Curve Analysis</h3>', unsafe_allow_html=True)

        # Full light curve with zoom/pan
        fig_full = create_interactive_lightcurve(current)
        st.plotly_chart(fig_full, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
        
        # Transit selector
        # st.markdown("### 🔍 Transit Detail View")
        st.markdown('<h3 style="color: #FFFFFF; margin: 0;">🔍 Transit Detail View</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        
        with col2:
            transit_num = st.selectbox(
                "Select Transit",
                range(len(current['transit_times'])),
                format_func=lambda x: f"Transit {x+1}"
            )
        
        # Zoomed transit view
        fig_zoom = create_transit_zoom(current, transit_num)
        st.plotly_chart(fig_zoom, use_container_width=True, config={'displayModeBar': False})
        
        # Physical Parameters
        # st.markdown("### 🔬 Physical Parameters")
        st.markdown('<h3 style="color: #FFFFFF; margin: 0;">🔬 Physical Parameters</h3>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        params = [
            ("Orbital Period", f"{current['period']:.2f} days", col1),
            ("Transit Depth", f"{current['depth']:.2f}%", col2),
            ("Duration", f"{current['duration']:.2f} hrs", col3),
            ("Radius Ratio (Rp/R*)", f"{current['radius_ratio']:.3f}", col4)
        ]
        
        for label, value, col in params:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div style='color: #a0a0a0; font-size: 0.85rem;'>{label}</div>
                    <div style='color: #667eea; font-size: 1.3rem; font-weight: bold; margin-top: 0.5rem;'>{value}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Vetting Actions
        # st.markdown("### 🎯 Your Vetting Decision")
        st.markdown('<h3 style="color: #FFFFFF; margin: 0;">🎯 Your Vetting Decision</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("👈 FALSE POSITIVE", use_container_width=True, key="fp"):
                st.session_state.labels.append('FALSE_POSITIVE')
                st.session_state.candidate_index += 1
                st.success("✅ Marked as False Positive")
                time.sleep(0.3)
                st.rerun()
        
        with col2:
            if st.button("👉 PLANET CANDIDATE", use_container_width=True, key="candidate"):
                st.session_state.labels.append('CANDIDATE')
                st.session_state.candidate_index += 1
                st.info("✅ Marked as Candidate")
                time.sleep(0.3)
                st.rerun()
        
        with col3:
            if st.button("👆 CONFIRMED PLANET", use_container_width=True, key="confirmed", type="primary"):
                st.session_state.labels.append('CONFIRMED')
                st.session_state.candidate_index += 1
                st.balloons()
                st.success("✅ Confirmed as Planet!")
                time.sleep(0.3)
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Navigation
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("⬅️ Previous", use_container_width=True, disabled=(idx == 0)):
                st.session_state.candidate_index -= 1
                if len(st.session_state.labels) > 0:
                    st.session_state.labels.pop()
                st.rerun()
        
        with col2:
            if st.button("⏭️ Skip", use_container_width=True):
                st.session_state.candidate_index += 1
                st.rerun()
        
        with col3:
            if st.button("🏠 Home", use_container_width=True):
                st.session_state.page = 'home'
                st.rerun()

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #667eea; padding: 2rem;'>
    <p style='font-size: 0.9rem; opacity: 0.6;'>
        🌌 EXOPLANET HUNTER v2.1 | Professional Vetting Platform
    </p>
</div>
""", unsafe_allow_html=True)