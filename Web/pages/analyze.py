import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# --- SHARED FUNCTIONS, STYLES, AND HEADER ---
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
        .candidate-card {
            background: linear-gradient(135deg, rgba(22, 33, 62, 0.95), rgba(15, 52, 96, 0.95));
            border: 2px solid rgba(102, 126, 234, 0.3); border-radius: 20px;
            padding: 2rem; margin: 1.5rem 0;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37); backdrop-filter: blur(10px);
        }
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none;
            border-radius: 15px; padding: 1rem 2rem; font-size: 1.1rem; font-weight: 600;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); transition: all 0.3s ease;
        }
        .stButton > button:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6); }
    </style>""", unsafe_allow_html=True)


# --- APP LAYOUT ---
apply_theme()
render_header()

st.title("📂 Upload & Analyze Your Own Data")
st.markdown("---")
st.info("Upload a CSV file with light curve data. The file must contain 'time' and 'flux' columns.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # --- Data Validation ---
        if 'time' not in df.columns or 'flux' not in df.columns:
            st.error("Error: The uploaded CSV must contain 'time' and 'flux' columns. Please check your file.")
        else:
            st.success("File uploaded and validated successfully!")
            
            with st.expander("Preview Your Data", expanded=True):
                st.dataframe(df.head())

            st.markdown("### 📈 Your Light Curve")
            
            # --- Plotting ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['time'],
                y=df['flux'],
                mode='lines',
                name='Uploaded Flux',
                line=dict(color='#4facfe', width=1.5),
                hovertemplate='Time: %{x:.4f}<br>Flux: %{y:.6f}<extra></extra>'
            ))
            fig.update_layout(
                title={'text': 'Interactive Light Curve of Your Data', 'font': {'size': 16, 'color': '#667eea'}},
                xaxis_title='Time',
                yaxis_title='Flux',
                height=500,
                paper_bgcolor='rgba(10, 14, 39, 0.8)',
                plot_bgcolor='rgba(22, 33, 62, 0.6)',
                font=dict(color='#e0e0e0', size=12),
                hovermode='x unified',
                dragmode='pan',
                xaxis=dict(gridcolor='rgba(102, 126, 234, 0.2)'),
                yaxis=dict(gridcolor='rgba(102, 126, 234, 0.2)')
            )
            fig.update_xaxes(fixedrange=False)
            fig.update_yaxes(fixedrange=False)
            
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 🤖 AI Analysis (Placeholder)")
            st.warning("Note: The AI analysis feature for custom uploads is currently in development. This is a demonstration of the data visualization pipeline.")
            
            if st.button("Run Mock Analysis"):
                with st.spinner("Analyzing..."):
                    time.sleep(2)
                    st.success("Mock analysis complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Predicted Period (days)", f"{np.random.uniform(5, 20):.2f}")
                    col2.metric("Predicted Depth (%)", f"{np.random.uniform(0.1, 1.5):.2f}")
                    col3.metric("AI Confidence", f"{np.random.uniform(0.6, 0.9):.1%}")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

# --- FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: #667eea; padding: 2rem;'><p style='font-size: 0.9rem; opacity: 0.6;'>🌌 EXOPLANET HUNTER v2.1 | Professional Vetting Platform</p></div>", unsafe_allow_html=True)
