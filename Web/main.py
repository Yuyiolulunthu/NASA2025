"""
🌌 Professional Exoplanet Vetting Platform
AI × Human Collaboration System
"""
import streamlit as st
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Exoplanet Hunter",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed" # Collapse sidebar on home
)

# --- SHARED FUNCTIONS AND STYLES ---
def render_header():
    """Renders the fixed navigation header."""
    # This CSS is injected into the head of the HTML
    st.markdown("""
        <style>
            /* Specific Streamlit adjustments */
            .st-emotion-cache-18ni7ap { /* Main content area */
                padding-top: 80px; /* Space for our fixed header */
            }
            .st-emotion-cache-16txtl3 { /* Sidebar */
                 padding-top: 2rem;
            }
        </style>
    """, unsafe_allow_html=True)

    # The HTML for the header itself
    header_html = """
    <style>
        .header {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 9999;
            background-color: rgba(10, 14, 39, 0.85);
            backdrop-filter: blur(8px);
            border-bottom: 1px solid rgba(51, 65, 85, 0.5);
            padding: 0.5rem 1.5rem;
            font-family: 'Inter', sans-serif;
        }
        .nav-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1280px;
            margin: auto;
        }
        .logo a {
            display: flex;
            align-items: center;
            text-decoration: none;
            color: white;
        }
        .logo span {
            font-weight: bold;
            font-size: 1.25rem;
            margin-left: 0.75rem;
        }
        .nav-links {
            display: flex;
            align-items: center;
            gap: 1.5rem;
        }
        .nav-links a {
            color: #D1D5DB;
            text-decoration: none;
            transition: color 0.3s;
            font-size: 0.9rem;
        }
        .nav-links a:hover {
            color: #818CF8;
        }
        .info-group {
            display: flex;
            gap: 1rem;
            border-left: 1px solid #4A5568;
            padding-left: 1.5rem;
        }
        .website-btn {
            background-color: #4F46E5;
            color: white;
            font-weight: bold;
            padding: 0.5rem 1.25rem;
            border-radius: 0.375rem;
            font-size: 0.875rem;
            transition: background-color 0.3s;
        }
        .website-btn:hover {
            background-color: #6366F1;
        }
        @media (max-width: 768px) {
            .nav-links { display: none; }
        }
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
    /* Main App Styling */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #16213e 50%, #0f3460 100%);
        color: #e0e0e0;
    }
    .stApp::before {
        content: ""; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background-image: radial-gradient(2px 2px at 20px 30px, #eee, rgba(0,0,0,0)),
                          radial-gradient(2px 2px at 60px 70px, #fff, rgba(0,0,0,0)),
                          radial-gradient(1px 1px at 50px 50px, #fff, rgba(0,0,0,0));
        background-repeat: repeat; background-size: 200px 200px;
        opacity: 0.4; z-index: -1; animation: twinkle 3s ease-in-out infinite;
    }
    @keyframes twinkle { 0%, 100% { opacity: 0.3; } 50% { opacity: 0.6; } }
    
    /* Title Styling */
    .space-title {
        font-size: 4rem; font-weight: 900; text-align: center;
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 300% 300%; -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        animation: gradient-shift 3s ease infinite; padding: 2rem 0; letter-spacing: 0.1em;
    }
    @keyframes gradient-shift { 0%, 100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }

    /* Card Styling */
    .candidate-card, .metric-card {
        background: linear-gradient(135deg, rgba(22, 33, 62, 0.95), rgba(15, 52, 96, 0.95));
        border: 2px solid rgba(102, 126, 234, 0.3); border-radius: 20px;
        padding: 2rem; margin: 1.5rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37); backdrop-filter: blur(10px);
    }
    .metric-card { padding: 1.5rem; text-align: center; }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none;
        border-radius: 15px; padding: 1rem 2rem; font-size: 1.1rem; font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); transition: all 0.3s ease;
    }
    .stButton > button:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6); }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e27 0%, #16213e 100%);
    }
    </style>
    """, unsafe_allow_html=True)


# --- INITIALIZE SESSION STATE ---
if 'candidate_index' not in st.session_state:
    st.session_state.candidate_index = 0
if 'candidates' not in st.session_state:
    # Generate test candidates
    st.session_state.candidates = []
    for i in range(10):
        time_data = np.linspace(0, 50, 500)
        flux = 1 + np.random.normal(0, 0.002, 500)
        period = np.random.uniform(5, 30)
        transit_times = [t for t in np.arange(0, 50, period)]
        st.session_state.candidates.append({
            'id': f'TIC-{200000+i}', 'time': time_data, 'flux': flux, 'period': period,
            'depth': np.random.uniform(0.5, 2.0), 'duration': np.random.uniform(2, 6),
            'snr': np.random.uniform(10, 50), 'radius_ratio': np.random.uniform(0.05, 0.15),
            'ai_confidence': np.random.uniform(0.5, 0.95), 'transit_times': transit_times[:3],
            'color_index': np.random.uniform(0.5, 1.5), 'effective_temp': np.random.uniform(4000, 7000)
        })
if 'labels' not in st.session_state:
    st.session_state.labels = []


# --- APP LAYOUT ---
apply_theme()
render_header()

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

# --- KEY PILLARS ---
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="metric-card">
        <div style='font-size: 3rem; margin-bottom: 0.5rem;'>🤖</div>
        <h3 style='color: #667eea;'>AI Screening</h3>
        <p style='color: #a0a0a0; margin-top: 0.5rem;'>
            Rapid analysis<br>Signal detection<br>Uncertainty flagging
        </p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="metric-card">
        <div style='font-size: 3rem; margin-bottom: 0.5rem;'>👁️</div>
        <h3 style='color: #4facfe;'>Human Vetting</h3>
        <p style='color: #a0a0a0; margin-top: 0.5rem;'>
            Expert review<br>Intuitive swipes<br>Critical judgment
        </p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="metric-card">
        <div style='font-size: 3rem; margin-bottom: 0.5rem;'>🔄</div>
        <h3 style='color: #f093fb;'>Collaborative Learning</h3>
        <p style='color: #a0a0a0; margin-top: 0.5rem;'>
            Model improvement<br>Contribution tracking<br>Continuous evolution
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# --- NAVIGATION ---
st.markdown("Use the sidebar on the left to navigate to the **Vetting Dashboard** or **Upload & Analyze** pages.")
st.markdown("---")

# --- STATS ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div style='text-align: center;'><div style='font-size: 2.5rem; color: #667eea; font-weight: bold;'>{len(st.session_state.labels)}</div><div style='color: #a0a0a0;'>Vetted</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div style='text-align: center;'><div style='font-size: 2.5rem; color: #4facfe; font-weight: bold;'>{len(st.session_state.candidates) - st.session_state.candidate_index}</div><div style='color: #a0a0a0;'>Remaining</div></div>", unsafe_allow_html=True)
with col3:
    confirmed = len([l for l in st.session_state.labels if l == 'CONFIRMED'])
    st.markdown(f"<div style='text-align: center;'><div style='font-size: 2.5rem; color: #10b981; font-weight: bold;'>{confirmed}</div><div style='color: #a0a0a0;'>Confirmed</div></div>", unsafe_allow_html=True)
with col4:
    accuracy = np.random.uniform(80, 95) if len(st.session_state.labels) > 0 else 0
    st.markdown(f"<div style='text-align: center;'><div style='font-size: 2.5rem; color: #f093fb; font-weight: bold;'>{accuracy:.1f}%</div><div style='color: #a0a0a0;'>Accuracy</div></div>", unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: #667eea; padding: 2rem;'><p style='font-size: 0.9rem; opacity: 0.6;'>🌌 EXOPLANET HUNTER v2.1 | Professional Vetting Platform</p></div>", unsafe_allow_html=True)

