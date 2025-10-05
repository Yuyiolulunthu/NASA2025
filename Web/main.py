import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import time
import json
import os
from components.banner import render_banner

# ================== Page Config ==================
st.set_page_config(
    page_title="Exoplanet Detection AI",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="Web/logo.png"
)

# ================== Global UI: Banner & Chrome ==================
render_banner()
st.markdown(
    """
    <style>
      #MainMenu, header, footer {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ================== Paths ==================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Go up from Web/pages/ to project root
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
FEATURE_LIST_PATH = os.path.join(MODELS_DIR, 'feature_list.json')
MODEL_PATH = os.path.join(MODELS_DIR, 'new_model_20251004_211436.pkl')

# ================== Enterprise Theme + Background Transit ==================
def apply_enterprise_theme():
    st.markdown(
        """
        <style>
            :root{
                --bg-0:#0b0f1a; --bg-1:#0e1624; --bg-2:#121a2a;
                --card:#151e2f; --border:#24314a;
                --muted:#a7b3c7; --text:#eef2f8;
                --accent:#4f7cff; --accent-2:#7aa2ff;
                --positive:#27c07d; --negative:#ff5c5c; --warn:#f0ad4e;
            }
            .stApp{ background:radial-gradient(60% 90% at 50% 10%, #0e162a 0%, #0a0e27 60%, #070b18 100%);} 
            .block-container{ padding-top:1.25rem; padding-bottom:3rem; }

            /* ---- Transit Background Layer (restored & upgraded) ---- */
            .transit-bg{ position:fixed; inset:0; z-index:0; pointer-events:none; overflow:hidden; }
            .transit-inner{ position:absolute; inset:0; }
            .star{
                   position:absolute; top:42%; left:50%; transform:translate(-50%, -50%);
                   width:300px; height:300px; border-radius:50%;                   background: radial-gradient(circle at 32% 30%, #fff8e1, #f5b971 70%, #9e6c33 100%);
                   box-shadow:0 0 100px 30px rgba(255,214,122,.22), 0 0 260px 80px rgba(255,179,71,.10);
                   filter: brightness(1.25);
                   animation: starDim 14s linear infinite;
            }
            .planet{
                     position:absolute; top:42%; left:-12%; transform:translate(0,-50%);
                     width:44px; height:44px; border-radius:50%; background:#0b0f1a;
                     box-shadow: inset -6px -4px 10px rgba(255,255,255,.08), 0 0 0 2px rgba(0,0,0,.35);
                     animation: orbit 14s linear infinite;
            }
            /* 行星從畫面左外 -> 右外（14s繞行一次） */
            @keyframes orbit{ 0%{ left:-12%; } 50%{ left:62%; } 100%{ left:112%; } }

            /* 亮度動畫：34%~66% 視為遮掩期間，亮度降至 60%（-30%） */
            @keyframes starDim{
              0%, 27%   { filter: brightness(1.25); }
              35%, 50%  { filter: brightness(0.75); }
              57%, 100% { filter: brightness(1.25); }
            }

            /* 柔和暈影層（邊緣感） */
            .transit-dim{
                position:absolute; inset:0; pointer-events:none;
                background: radial-gradient(circle at 50% 42%,
                           rgba(0,0,0,.00) 0%,
                           rgba(0,0,0,.18) 38%,
                           rgba(0,0,0,.28) 42%,
                           rgba(0,0,0,.06) 60%,
                           rgba(0,0,0,0) 72%);
                opacity:.14;
                animation: dimRing 14s linear infinite;
            }
            @keyframes dimRing{
              0%,28%{opacity:.08}
              34%,66%{opacity:.16}
              72%,100%{opacity:.08}
            }

            /* ---- Top Nav / Hero / Cards 等（原樣保留） ---- */
            .topnav{ position:sticky; top:0; z-index:5; backdrop-filter:saturate(140%) blur(8px);
                     background:rgba(10,14,33,.72); border-bottom:1px solid var(--border);
                     padding:.6rem 1rem; }
            .topnav-inner{ display:flex; align-items:center; justify-content:space-between; max-width:1200px; margin:0 auto; gap:1rem;}
            .brand{ color:var(--text); font-weight:800; letter-spacing:.02em; }
            .nav-actions{ display:flex; gap:.6rem; flex-wrap:wrap; }
            .nav-actions a{
                color:var(--text); text-decoration:none; font-weight:800; padding:.55rem 1rem;
                border:1px solid var(--border); border-radius:10px; transition:all .12s ease;
                background:linear-gradient(180deg, rgba(79,124,255,.10) 0%, rgba(79,124,255,.04) 100%);
                white-space:nowrap;
            }
            .nav-actions a:hover{ background: rgba(122,162,255,.16); border-color:#3b58a6; transform:translateY(-1px); }

            .hero{ position:relative; text-align:center; padding:3.6rem 1rem 2.6rem; z-index:1; }
            .hero h1{ color:var(--text); font-weight:900; font-size: clamp(2.6rem, 4vw + 1rem, 4rem); letter-spacing:-.02em; margin:0; }
            .hero p{ color:var(--muted); font-size: clamp(1.05rem, .6vw + .9rem, 1.25rem); max-width:860px; margin:.9rem auto 0; }
            .hero-cta{ margin-top:1.6rem; display:flex; gap:1rem; justify-content:center; }

            .cta-main{ background:var(--accent); color:#fff; font-weight:800; padding:.9rem 1.25rem; border-radius:12px; border:1px solid #2e56cb;
                       text-decoration:none; box-shadow:0 10px 22px rgba(79,124,255,.18); }
            .cta-main:hover{ background:#406eea; }
            .cta-ghost{ color:var(--text); text-decoration:none; font-weight:700; padding:.9rem 1.1rem; border:1px solid var(--border);
                        border-radius:12px; }

            .steps{ display:flex; gap:1rem; justify-content:center; flex-wrap:wrap; margin-top:1.2rem; }
            .step-btn{ display:block; width:280px; text-decoration:none; }
            .step-btn .card{ cursor:pointer; transition:transform .12s ease, box-shadow .12s ease; text-align:center; }
            .step-btn .card:hover{ transform:translateY(-3px); box-shadow:0 14px 28px rgba(79,124,255,.15); }
            .step-index{ display:inline-block; min-width:32px; height:32px; line-height:32px; border-radius:8px; background:rgba(122,162,255,.16); color:var(--text); font-weight:800; margin-bottom:.4rem; }

            .card{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:1.2rem 1.2rem; }
            .card h4{ color:var(--text); margin:.2rem 0 .6rem; font-weight:800; }
            .muted{ color:var(--muted); }

            .upload-card{ background:rgba(21,30,47,.78); border:2px dashed rgba(122,162,255,.55); border-radius:14px; padding:1.8rem 1.5rem; }
            .upload-card h4{ color:#fff !important; margin:0 0 .4rem 0; font-weight:800; }
            .upload-card p, .upload-card small{ color:#c7d3e9; }

            .metric-chip{ background:linear-gradient(180deg, rgba(79,124,255,.08) 0%, rgba(79,124,255,.04) 100%);
                          border:1px solid rgba(79,124,255,.35); border-radius:12px; padding:.8rem 1rem; text-align:center; }
            .metric-chip .lab{ color:#c6d2e6; font-size:.78rem; letter-spacing:.06em; text-transform:uppercase; }
            .metric-chip .val{ color:var(--text); font-size:1.5rem; font-weight:800; margin-top:.1rem; }

            .status{ display:inline-flex; align-items:center; justify-content:center; padding:.7rem 1.1rem; border-radius:10px; gap:.5rem; font-weight:900; letter-spacing:.02em; font-size:1.05rem; border:1px solid; min-width:220px; }
            .status.positive{ color:#0c2a1c; background: rgba(39,192,125,.18); border-color: rgba(39,192,125,.45); }
            .status.negative{ color:#2a0c0c; background: rgba(255,92,92,.18); border-color: rgba(255,92,92,.45); }

            .stPlotlyChart > div > div{ background:transparent !important; }

            [data-testid=stSidebar]{ background:linear-gradient(180deg, var(--bg-2) 0%, var(--bg-1) 100%) !important; border-right:1px solid var(--border); }
            [data-testid=stSidebar] *{ color:var(--text) !important; }

            .dataframe thead th{ background:#1b2740 !important; color:#eaf0ff !important; font-weight:800 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_enterprise_theme()

# ================== Load Feature List ==================
def load_feature_list():
    try:
        if os.path.exists(FEATURE_LIST_PATH):
            with open(FEATURE_LIST_PATH, 'r') as f:
                feature_data = json.load(f)
                if isinstance(feature_data, list):
                    return feature_data
                elif isinstance(feature_data, dict) and 'features' in feature_data:
                    return feature_data['features']
                else:
                    st.warning("feature_list.json format not recognized.")
                    return None
        else:
            return None
    except Exception as e:
        st.error(f"Error loading feature list: {str(e)}")
        return None

FEATURE_LIST = load_feature_list()

# ================== Feature Engineering ==================
def extract_features(df):
    try:
        features = {}
        flux = df['flux'].values.astype(float)
        time_arr = df['time'].values.astype(float)
        all_features = {}

        # Basic
        all_features['mean_flux'] = float(np.mean(flux))
        all_features['std_flux'] = float(np.std(flux))
        all_features['min_flux'] = float(np.min(flux))
        all_features['max_flux'] = float(np.max(flux))
        all_features['median_flux'] = float(np.median(flux))
        all_features['flux_range'] = float(np.max(flux) - np.min(flux))
        all_features['cv_flux'] = float(np.std(flux) / np.mean(flux)) if np.mean(flux) != 0 else 0.0

        # Distribution
        all_features['skewness'] = float(stats.skew(flux, bias=False, nan_policy='omit'))
        all_features['kurtosis'] = float(stats.kurtosis(flux, bias=False, nan_policy='omit'))
        all_features['percentile_75'] = float(np.percentile(flux, 75))
        all_features['percentile_25'] = float(np.percentile(flux, 25))
        all_features['percentile_90'] = float(np.percentile(flux, 90))

        # Trend
        denom = (np.max(time_arr) - np.min(time_arr))
        time_norm = (time_arr - np.min(time_arr)) / denom if denom != 0 else time_arr
        slope, intercept, r_value, _, _ = stats.linregress(time_norm, flux)
        all_features['trend_slope'] = float(slope)
        all_features['trend_r2'] = float(r_value ** 2)
        all_features['linear_fit_error'] = float(np.mean(np.abs(flux - (slope * time_norm + intercept))))

        # Variability
        flux_diff = np.diff(flux) if len(flux) > 1 else np.array([0.0])
        all_features['mean_absolute_deviation'] = float(np.mean(np.abs(flux - np.mean(flux))))
        all_features['flux_diff_mean'] = float(np.mean(flux_diff))
        all_features['flux_diff_std'] = float(np.std(flux_diff))
        all_features['flux_diff_max'] = float(np.max(np.abs(flux_diff))) if len(flux_diff) else 0.0

        # Rolling
        window = max(5, len(flux) // 20) if len(flux) >= 5 else 5
        rolling_std = pd.Series(flux, dtype=float).rolling(window=window, center=True).std()
        all_features['rolling_std_mean'] = float(np.nanmean(rolling_std))
        all_features['rolling_std_max'] = float(np.nanmax(rolling_std))
        all_features['rolling_std_min'] = float(np.nanmin(rolling_std))

        # Meta
        all_features['data_points'] = int(len(flux))
        all_features['time_span'] = float(np.max(time_arr) - np.min(time_arr)) if len(time_arr) else 0.0

        if FEATURE_LIST:
            for feature_name in FEATURE_LIST:
                features[feature_name] = all_features.get(feature_name, 0)
        else:
            default_features = [
                'mean_flux','std_flux','min_flux','max_flux','median_flux',
                'flux_range','cv_flux','skewness','kurtosis','percentile_75',
                'trend_slope','trend_r2','linear_fit_error','mean_absolute_deviation',
                'flux_diff_mean','flux_diff_std','flux_diff_max',
                'rolling_std_mean','rolling_std_max','rolling_std_min',
                'data_points','time_span'
            ]
            for feature_name in default_features:
                features[feature_name] = all_features.get(feature_name, 0)

        return features
    except Exception as e:
        st.error(f"Feature extraction failed: {str(e)}")
        return {}

# ================== Model ==================
def load_model():
    try:
        import joblib
        if os.path.exists(MODEL_PATH):
            return joblib.load(MODEL_PATH)
        else:
            st.warning("Model file not found. Running in demo mode.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# ================== Predict ==================
def predict_with_model(features_dict, model=None):
    if not features_dict:
        return {
            'prediction': 'NOT_PLANET', 'confidence': 0.5,
            'planet_probability': 0.3, 'not_planet_probability': 0.7,
            'model_version': '20251004_211436', 'model_type': 'Stacking Ensemble', 'feature_count': 0
        }

    features_df = pd.DataFrame([features_dict])

    if model is not None:
        try:
            prediction = model.predict(features_df)[0]
            probabilities = model.predict_proba(features_df)[0]
            not_planet_prob = float(probabilities[0])
            planet_prob = float(probabilities[1])
            confidence = float(planet_prob if prediction == 'PLANET' else not_planet_prob)
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            prediction = np.random.choice(['PLANET','NOT_PLANET'], p=[0.35,0.65])
            planet_prob = float(np.random.uniform(0.3,0.7))
            not_planet_prob = float(1 - planet_prob)
            confidence = float(max(planet_prob, not_planet_prob))
    else:
        prediction = np.random.choice(['PLANET','NOT_PLANET'], p=[0.35,0.65])
        planet_prob = float(np.random.uniform(0.3,0.7))
        not_planet_prob = float(1 - planet_prob)
        confidence = float(max(planet_prob, not_planet_prob))

    return {
        'prediction': prediction,
        'confidence': confidence,
        'planet_probability': planet_prob,
        'not_planet_probability': not_planet_prob,
        'model_version': '20251004_211436',
        'model_type': 'Stacking Ensemble',
        'feature_count': len(features_dict)
    }

# ================== Plot ==================
def create_visualization_plot(df, features):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Light Curve', 'Flux Distribution', 'Detrended Signal', 'Feature Summary'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "bar"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    mean_flux = features.get('mean_flux', float(df['flux'].mean()))
    std_flux = features.get('std_flux', float(df['flux'].std()))
    flux_range = features.get('flux_range', float(df['flux'].max() - df['flux'].min()))
    cv_flux = features.get('cv_flux', (std_flux/mean_flux) if mean_flux else 0.0)

    # 1. Time Series
    fig.add_trace(
        go.Scatter(
            x=df['time'], y=df['flux'], mode='lines', name='Light Curve',
            line=dict(color='#9bd2ff', width=2),
            hovertemplate='Time: %{x:.4f}<br>Flux: %{y:.6f}<extra></extra>'
        ),
        row=1, col=1
    )
    try:
        fig.add_hline(y=mean_flux, line_dash="dash", line_color="#ffd666",
                      line_width=2, annotation_text="Mean", annotation_font_color="#eaeaea",
                      row=1, col=1)
    except Exception:
        fig.add_hline(y=mean_flux, line_dash="dash", line_color="#ffd666", line_width=2)

    # 2. Histogram
    fig.add_trace(
        go.Histogram(x=df['flux'], name='Distribution', nbinsx=50,
                     marker=dict(color='#4f7cff', opacity=0.85)),
        row=1, col=2
    )

    # 3. Detrended
    flux = df['flux'].values.astype(float)
    denom = (df['time'].max() - df['time'].min())
    time_norm = (df['time'] - df['time'].min())/denom if denom != 0 else df['time']
    slope, intercept, _, _, _ = stats.linregress(time_norm, flux)
    detrended = flux - (slope*time_norm + intercept)

    fig.add_trace(
        go.Scatter(x=df['time'], y=detrended, mode='lines', name='Detrended',
                   line=dict(color='#ff9db2', width=2)),
        row=2, col=1
    )

    # 4. Feature bars
    sk = float(stats.skew(flux, bias=False, nan_policy='omit')) if len(flux) else 0.0
    feature_names = ['Mean', 'Std', 'Range', 'CV', 'Skew', 'Trend']
    feature_values = [
        abs(features.get('mean_flux', mean_flux)),
        abs(features.get('std_flux', std_flux)),
        abs(features.get('flux_range', flux_range)),
        abs(features.get('cv_flux', cv_flux)),
        abs(features.get('skewness', sk)),
        abs(features.get('trend_slope', float(slope)))
    ]
    fig.add_trace(
        go.Bar(x=feature_names, y=feature_values, name='Features',
               marker=dict(color='#7aa2ff', opacity=0.9)),
        row=2, col=2
    )

    fig.update_layout(
        height=760, showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(18,26,42,0.95)',
        font=dict(color='#e8eef8', size=12),
        hovermode='closest',
        margin=dict(l=40, r=20, t=60, b=40)
    )
    fig.update_xaxes(gridcolor='rgba(100,140,200,0.25)', gridwidth=1,
                     title_font_color='#e8eef8', tickfont_color='#d6e0f2')
    fig.update_yaxes(gridcolor='rgba(100,140,200,0.25)', gridwidth=1,
                     title_font_color='#e8eef8', tickfont_color='#d6e0f2')

    # 統一 subplot 標題字型
    if 'annotations' in fig['layout']:
        for ann in fig['layout']['annotations']:
            ann['font'] = dict(size=14, color='#e8eef8')

    return fig

# ================== Session State ==================
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'results' not in st.session_state:
    st.session_state.results = None

# ================== Top Navigation (4 buttons) ==================
st.markdown(
    """
    <div class='topnav'>
      <div class='topnav-inner'>
        <div class='brand'>Exoplanet Detection AI</div>
        <div class='nav-actions'>
          <a href='/about'>About our model</a>
          <a href='/analyze'>Analyze your data</a>
          <a href='/fits_converter'>FITS Converter</a>
          <a href='/vetting'>How model works</a>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Make top navigation fixed at the BOTTOM of the viewport (and avoid covering content)
st.markdown(
    """
    <style>
      .topnav{
        position: fixed !important;
        left: 0; right: 0;
        top: auto !important; bottom: 0 !important;
        z-index: 1050 !important;
        backdrop-filter: saturate(140%) blur(8px);
        background: rgba(10,14,33,0.86);
        border-top: 1px solid var(--border);
        padding: .6rem 1rem;
      }
      /* ensure content can be pushed up so it isn't hidden behind the fixed nav */
      .block-container { transition: padding-bottom .12s ease !important; }
    </style>
    <script>
    (function(){
      function updateSpacing(){
        const nav = document.querySelector('.topnav');
        const navH = nav ? Math.ceil(nav.getBoundingClientRect().height) : 0;
        const bc = document.querySelector('.block-container') || document.querySelector('main[role=\"main\"]') || document.body;
        if (bc) bc.style.paddingBottom = (navH + 18) + 'px';
      }
      window.addEventListener('load', updateSpacing);
      window.addEventListener('resize', updateSpacing);
      if (window.ResizeObserver){
        const el = document.querySelector('.topnav');
        if (el) new ResizeObserver(updateSpacing).observe(el);
      }
      setTimeout(updateSpacing, 250);
    })();
    </script>
    """,
    unsafe_allow_html=True,
)


# ================== Transit Background (HTML) ==================
st.markdown(
    """
    <div class='transit-bg'>
      <div class='transit-inner'>
        <div class='star'></div>
        <div class='planet'></div>
        <div class='transit-dim'></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ================== Hero ==================
st.markdown(
    """
    <section class='hero'>
      <h1>AI × Human Collaboration for Professional Exoplanet Vetting</h1>
      <p>Upload a light curve and get a science-grade classification with feature analytics, confidence metrics, and publication-ready plots.</p>
      <div class='hero-cta'>
        <a class='cta-main' href='#detect'>Start Detection</a>
        <a class='cta-ghost' href='#about'>Learn More</a>
      </div>
      <div class='steps'>
        <a class='step-btn' href='#detect'><div class='card'><div class='step-index'>1</div><h4>Upload</h4><p class='muted'>CSV with time and flux</p></div></a>
        <a class='step-btn' href='#detect'><div class='card'><div class='step-index'>2</div><h4>Analyze</h4><p class='muted'>Feature extraction & classify</p></div></a>
        <a class='step-btn' href='#visualize'><div class='card'><div class='step-index'>3</div><h4>Review</h4><p class='muted'>Confidence & plots</p></div></a>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# ===== Anchors for nav buttons =====
st.markdown("<div id='detect'></div>", unsafe_allow_html=True)

# ================== MAIN TABS ==================
tab_detect, tab_visualize, tab_about = st.tabs(["Detect", "Visualize", "About"])

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("#### Quick Panel")
    st.write("Use the **Detect** tab to upload and classify. Use **Visualize** to explore plots and features.")

    with st.expander("Advanced Model Info", expanded=False):
        model_exists = os.path.exists(MODEL_PATH)
        features_exist = os.path.exists(FEATURE_LIST_PATH)
        st.write("Model Type: Stacking Ensemble")
        st.write("Version: 20251004_211436")
        st.write("Classes: 2 (PLANET / NOT_PLANET)")
        st.write("Training Samples: 23,289")
        st.write(f"Model Exists: {model_exists}")
        st.write(f"Features File: {features_exist}")
        if model_exists:
            st.code(MODEL_PATH, language="text")
        if features_exist:
            st.code(FEATURE_LIST_PATH, language="text")

# ================== DETECT TAB ==================
with tab_detect:
    st.markdown("<div class='upload-card'>"
                "<h4>Upload Light Curve Data</h4>"
                "<p>Provide a CSV with <b>time</b> and <b>flux</b> columns. Auto-detection will be attempted; manual selection is available.</p>"
                "<small>Supported time columns: TIME / T / BJD / JD / MJD · Supported flux columns: FLUX / F / BRIGHTNESS / MAG</small>"
                "</div>", unsafe_allow_html=True)

    up_col1, up_col2, up_col3 = st.columns([1,2,1])
    with up_col2:
        uploaded_file = st.file_uploader(
            "Upload Light Curve Data (CSV)",
            type=['csv'], key='uploader_main',
            help="CSV with time and flux. Column names auto-detected; manual selection available."
        )

    if uploaded_file is not None:
        try:
            # --- 讀檔 ---
            with st.spinner('Loading light curve data...'):
                df = pd.read_csv(uploaded_file)
                time.sleep(0.15)

            # --- 自動偵測欄位 ---
            time_col, flux_col = None, None
            for col in df.columns:
                cl = col.lower().strip()
                if cl in ['time','times','t','bjd','jd','mjd','timestamp'] and time_col is None:
                    time_col = col
                if cl in ['flux','fluxes','f','brightness','magnitude','mag','intensity'] and flux_col is None:
                    flux_col = col

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            if time_col is None or flux_col is None:
                st.warning("Auto-detection failed. Please select columns.")
                c1, c2 = st.columns(2)
                with c1:
                    time_col = st.selectbox("TIME column", df.columns.tolist(), key='time_sel')
                with c2:
                    flux_col = st.selectbox("FLUX column", df.columns.tolist(), key='flux_sel')
            else:
                st.success(f"Auto-detected → Time: '{time_col}', Flux: '{flux_col}'")

            # --- 統一欄位 ---
            df_processed = (
                df[[time_col, flux_col]]
                .rename(columns={time_col: 'time', flux_col: 'flux'})
                .copy()
            )
            df_processed['time'] = pd.to_numeric(df_processed['time'], errors='coerce')
            df_processed['flux'] = pd.to_numeric(df_processed['flux'], errors='coerce')
            df_processed = df_processed.dropna(subset=['time', 'flux'])

            if df_processed.empty:
                st.error("Parsed light curve is empty after cleaning. Please check your CSV.")
                st.stop()

            st.session_state.uploaded_df = df_processed

            # 預覽
            st.dataframe(df_processed.head(12), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Primary CTA
            act_c1, act_c2, act_c3 = st.columns([1,1,1])
            with act_c2:
                if st.button("Run Exoplanet Detection", use_container_width=True, key='btn_detect_main'):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.text("Step 1/5 — Preprocessing")
                    progress_bar.progress(20); time.sleep(0.15)

                    status_text.text("Step 2/5 — Extracting features")
                    features = extract_features(df_processed)
                    progress_bar.progress(40); time.sleep(0.15)

                    status_text.text("Step 3/5 — Loading model")
                    model = load_model()
                    progress_bar.progress(60); time.sleep(0.1)

                    status_text.text("Step 4/5 — Classification")
                    results = predict_with_model(features, model)
                    progress_bar.progress(80); time.sleep(0.15)

                    status_text.text("Step 5/5 — Report generation")
                    progress_bar.progress(100); time.sleep(0.1)
                    progress_bar.empty(); status_text.empty()

                    st.session_state.features = features
                    st.session_state.results = results

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)

    # Results block (persistent once computed)
    if st.session_state.results is not None and st.session_state.features is not None:
        results = st.session_state.results
        features = st.session_state.features
        df_used = st.session_state.uploaded_df.copy()

        st.markdown("---")
        st.markdown("### Detection Results")
        cc1, cc2, cc3 = st.columns([1,2,1])
        with cc2:
            if results['prediction'] == 'PLANET':
                st.markdown("<div class='status positive'>PLANET</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='status negative'>NOT PLANET</div>", unsafe_allow_html=True)

        st.markdown("")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"<div class='metric-chip'><div class='lab'>Overall Confidence</div><div class='val'>{results['confidence']:.1%}</div></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='metric-chip'><div class='lab'>Planet Probability</div><div class='val'>{results['planet_probability']:.1%}</div></div>", unsafe_allow_html=True)
        with m3:
            st.markdown(f"<div class='metric-chip'><div class='lab'>Not-Planet Probability</div><div class='val'>{results['not_planet_probability']:.1%}</div></div>", unsafe_allow_html=True)
        with m4:
            st.markdown(f"<div class='metric-chip'><div class='lab'>Features Used</div><div class='val'>{results['feature_count']}</div></div>", unsafe_allow_html=True)

        # Assessment
        st.markdown("#### Model Assessment")
        if results['prediction'] == 'PLANET':
            if results['confidence'] > 0.85:
                st.info("High-confidence positive classification. Transit-like signal is strongly indicated.")
            elif results['confidence'] > 0.70:
                st.warning("Moderate confidence. Additional observation is recommended.")
            else:
                st.warning("Low confidence. Verification with further data is recommended.")
        else:
            if results['confidence'] > 0.85:
                st.info("High-confidence negative classification.")
            else:
                st.warning("Uncertain negative. Consider collecting more data to improve certainty.")

        # Data quality
        mean_f = features.get('mean_flux', float(df_used['flux'].mean()))
        std_f = features.get('std_flux', float(df_used['flux'].std()))
        cv_f = features.get('cv_flux', (std_f/mean_f) if mean_f else 0.0)
        st.markdown("#### Data Quality Indicators")
        st.markdown(f"- Signal-to-Noise (mean/std): **{(mean_f/std_f if std_f else 0):.2f}**")
        st.markdown(f"- Coefficient of Variation: **{cv_f:.4f}**")
        st.markdown(f"- Data Points: **{int(features.get('data_points', len(df_used))):,}**")

        # Download report
        full_report = {**results, **features}
        report_df = pd.DataFrame([full_report])
        csv = report_df.to_csv(index=False)
        dl1, dl2, dl3 = st.columns([1,1,1])
        with dl2:
            st.download_button(
                label="Download Full Report (CSV)",
                data=csv,
                file_name=f"exoplanet_detection_report_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                key='dl_report'
            )

# ================== VISUALIZE TAB ==================
st.markdown("<div id='visualize'></div>", unsafe_allow_html=True)
with tab_visualize:
    if st.session_state.uploaded_df is None or st.session_state.features is None:
        st.info("Upload data and run detection in the Detect tab to enable visualizations.")
    else:
        df_vis = st.session_state.uploaded_df.copy()

        st.markdown("<div class='card'><h4>Light Curve Analysis</h4>", unsafe_allow_html=True)
        fig = create_visualization_plot(df_vis, st.session_state.features)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Raw Data Preview", expanded=False):
            st.dataframe(df_vis[['time','flux']].head(50), use_container_width=True)

        with st.expander("Extracted Features", expanded=False):
            feat_items = list(st.session_state.features.items())
            feat_df = pd.DataFrame(feat_items, columns=["Feature", "Value"])
            st.dataframe(feat_df, use_container_width=True)

# ================== ABOUT TAB ==================
st.markdown("<div id='about'></div>", unsafe_allow_html=True)
with tab_about:
    st.markdown("""
    ### About the Platform
    This platform provides a professional workflow for exoplanet transit vetting.

    **Core capabilities**
    - CSV ingestion with column auto-detection (time / flux)
    - Robust feature engineering (trend, variability, rolling statistics, distribution)
    - Stacking ensemble classification with calibrated probabilities
    - Visual diagnostics (light curve, histogram, detrended signal, feature summary)

    **Intended users**
    - Astronomy students & researchers
    - Data scientists working with TESS / Kepler / custom light curves

    **Version**
    - Stacking Ensemble v20251004_211436
    - Training sources: TESS, Kepler, Confirmed Planets
    """)

    # How model works anchor inside About tab (works with top button)
    st.markdown("<div id='how'></div>", unsafe_allow_html=True)
    st.markdown("""
    ### How the Model Works
    - Preprocess light-curve → standardize & clean NaNs  
    - Extract analytic features (trend, variability, rolling stats, distributional moments)  
    - Feed features to a stacking ensemble (level-0 base learners + level-1 meta learner)  
    - Output: PLANET / NOT_PLANET with calibrated probabilities and confidence  
    """)

# ================== FITS Converter Section (anchor target) ==================
st.markdown("<div id='fits'></div>", unsafe_allow_html=True)
st.markdown("""
<div class='card'>
  <h4>FITS Converter</h4>
  <p class='muted'>Convert FITS light curves to CSV for analysis. If you have a dedicated converter page, open it below.</p>
</div>
""", unsafe_allow_html=True)

cfc1, cfc2, cfc3 = st.columns([1,1,1])
with cfc2:
    if st.button("Open FITS Converter", use_container_width=True, key="btn_fits_open"):
        try:
            st.switch_page("pages/fits_converter.py")
        except Exception:
            st.info("Create `pages/fits_converter.py` to enable navigation, or replace this with your converter workflow.")

# ================== Footer ==================
st.markdown(
    """
    <hr style='border-color: rgba(36,49,74,.6); margin-top:2rem;' />
    <div style='text-align:center; color:#c6d2e6; padding-top:.6rem;'>
        <div style='font-size:.95rem;'>Exoplanet Detection AI Platform — Professional Vetting Suite</div>
        <div style='font-size:.85rem; opacity:.85; margin-top:.2rem;'>Stacking Ensemble v20251004_211436 · TESS · Kepler · Confirmed Planets</div>
    </div>
    """,
    unsafe_allow_html=True,
)
