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

render_banner()
hide_streamlit_header_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_header_style, unsafe_allow_html=True)

# ================== Paths ==================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Go up from Web/pages/ to project root
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
FEATURE_LIST_PATH = os.path.join(MODELS_DIR, 'feature_list.json')
MODEL_PATH = os.path.join(MODELS_DIR, 'new_model_20251004_211436.pkl')

# ================== Premium (Enterprise) Theme ==================
def apply_enterprise_theme():
    st.markdown("""
    <style>
        :root{
            --bg-0:#0b0f1a;
            --bg-1:#0e1624;
            --bg-2:#121a2a;
            --card:#151e2f;
            --border:#24314a;
            --muted:#a7b3c7;
            --text:#eef2f8;
            --accent:#4f7cff;
            --accent-2:#7aa2ff;
            --positive:#27c07d;
            --negative:#ff5c5c;
            --warn:#f0ad4e;
        }

        /* Base */
        .stApp { background: linear-gradient(180deg, var(--bg-1) 0%, var(--bg-0) 100%); }
        #MainMenu, header, footer {visibility:hidden;}
        .block-container{ padding-top:2rem; padding-bottom:4rem; }

        /* Title */
        .main-title{
            color: var(--text);
            font-weight: 800;
            font-size: clamp(3.0rem, 4.4vw + 0.6rem, 4.2rem); /* 放大且自適應 */
            line-height: 1.12;
            letter-spacing: -0.02em;
            text-align:center;
            margin: 0 0 .35rem 0;
        }
        .subtitle{
            color: var(--muted);
            font-size: clamp(1.0rem, 0.6vw + 0.85rem, 1.2rem);
            text-align:center;
            margin-bottom: 1.6rem;
        }

        /* Global headings 層級微調 */
        h3 { font-size: 1.45rem; }
        h4 { font-size: 1.18rem; }

        /* Cards */
        .card{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.25rem 1.25rem;
        }
        .card.tight{ padding: 1rem 1rem; }
        .card h4{ color: var(--text); margin: 0 0 .75rem 0; font-weight:700; }

        /* Upload card with dashed border */
        .upload-card{
            background: rgba(21,30,47,0.75);
            border: 2px dashed rgba(122,162,255,0.55);
            border-radius: 12px;
            padding: 2.2rem 2rem;
            transition: border-color .12s ease, box-shadow .12s ease, background .12s ease;
        }
        .upload-card:hover{
            border-color: rgba(122,162,255,0.8);
            box-shadow: 0 10px 22px rgba(79,124,255,.18);
            background: rgba(21,30,47,0.85);
        }
        /* 讓 Upload Light Curve Data 標題變成純白 */
        .upload-card h4{
            color:#ffffff !important;
            margin-top:0;
            font-weight:700;
        }
        .upload-card p{ color:#c7d3e9; }
        .upload-card small{ color:#9fb0cf; }

        /* Metric grid */
        .metric-card{
            background: linear-gradient(180deg, rgba(79,124,255,0.08) 0%, rgba(79,124,255,0.04) 100%);
            border: 1px solid rgba(79,124,255,0.35);
            border-radius: 10px;
            padding: .9rem 1rem;
            text-align:center;
            transition: transform .12s ease;
        }
        .metric-card:hover{ transform: translateY(-2px); }
        .metric-label{ color:#c6d2e6; font-size:.8rem; letter-spacing:.06em; text-transform:uppercase; }
        .metric-value{ color: var(--text); font-size: 1.8rem; font-weight: 800; margin-top:.2rem; }

        /* Buttons */
        .stButton > button{
            width:100%;
            background: var(--accent);
            color: #fff;
            border: 1px solid #2e56cb;
            border-radius: 10px;
            padding: .9rem 1rem;
            font-weight: 700;
            letter-spacing:.02em;
            transition: background .12s ease, transform .12s ease, box-shadow .12s ease;
            box-shadow: 0 8px 20px rgba(79,124,255,.18);
        }
        .stButton > button:hover{
            background: #406eea;
            transform: translateY(-1px);
            box-shadow: 0 10px 22px rgba(79,124,255,.24);
        }

        /* Status chips (prediction) */
        .status{
            display:inline-flex; align-items:center; justify-content:center;
            padding:.6rem 1rem; border-radius: 8px; gap:.5rem;
            font-weight:800; letter-spacing:.02em; font-size:1.05rem;
            border:1px solid; min-width:220px;
        }
        .status.positive{ color:#0c2a1c; background: rgba(39,192,125,.18); border-color: rgba(39,192,125,.45); }
        .status.negative{ color:#2a0c0c; background: rgba(255,92,92,.18); border-color: rgba(255,92,92,.45); }

        /* Sidebar */
        [data-testid="stSidebar"]{
            background: linear-gradient(180deg, var(--bg-2) 0%, var(--bg-1) 100%) !important;
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] li{
            color: var(--text) !important;
        }

        /* Tables/Dataframe */
        .dataframe { color: var(--text) !important; }
        .dataframe thead th{
            background: #1b2740 !important;
            color: #eaf0ff !important;
            font-weight: 700 !important;
        }
        .dataframe tbody td{ color: #dbe5f8 !important; }

        /* Alerts (re-tone) */
        .stAlert{ background: rgba(27,39,64,.7) !important; border: 1px solid var(--border) !important; color: var(--text) !important; }

        /* Code blocks */
        .stCodeBlock{
            background: #10192c !important;
            border: 1px solid var(--border) !important;
        }
        code{ color:#9bd2ff !important; }

        /* Plotly container background alignment */
        .stPlotlyChart > div > div{
            background: transparent !important;
        }
    </style>
    """, unsafe_allow_html=True)

apply_enterprise_theme()

# ================== Feature List ==================
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
        flux = df['flux'].values
        time_arr = df['time'].values
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
        window = max(5, len(flux) // 20)
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
                'rolling_std_mean','rolling_std_max','rolling_std_min'
            ]
            for feature_name in default_features:
                features[feature_name] = all_features.get(feature_name, 0)

        return features
    except Exception as e:
        st.error(f"Feature extraction failed: {str(e)}")
        return {}

# ================== Model Loading ==================
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
            'prediction': 'NOT_PLANET',
            'confidence': 0.5,
            'planet_probability': 0.3,
            'not_planet_probability': 0.7,
            'model_version': '20251004_211436',
            'model_type': 'Stacking Ensemble',
            'feature_count': 0
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
    fig.add_hline(y=mean_flux, line_dash="dash", line_color="#ffd666",
                  line_width=2, annotation_text="Mean", annotation_font_color="#eaeaea",
                  row=1, col=1)

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
    feature_names = ['Mean', 'Std', 'Range', 'CV', 'Skew', 'Trend']
    feature_values = [
        abs(features.get('mean_flux', mean_flux)),
        abs(features.get('std_flux', std_flux)),
        abs(features.get('flux_range', flux_range)),
        abs(features.get('cv_flux', cv_flux)),
        abs(features.get('skewness', float(stats.skew(flux)))),
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

    for ann in fig['layout']['annotations']:
        ann['font'] = dict(size=14, color='#e8eef8')
    return fig

# ================== Header ==================
st.markdown('<div class="main-title">Exoplanet Detection AI Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning Classification · Stacking Ensemble · Operational Dashboard</div>', unsafe_allow_html=True)

# ================== Sidebar ==================
with st.sidebar:
    st.markdown("#### Model Information")
    model_exists = os.path.exists(MODEL_PATH)
    features_exist = os.path.exists(FEATURE_LIST_PATH)

    st.markdown('<div class="card tight">', unsafe_allow_html=True)
    st.write("Model Path:")
    st.code(MODEL_PATH, language="text")
    st.write("Features Path:")
    st.code(FEATURE_LIST_PATH, language="text")
    st.write(f"Model Exists: {model_exists}")
    st.write(f"Features Exist: {features_exist}")
    st.markdown('</div>', unsafe_allow_html=True)

    if model_exists:
        st.info("Model loaded.")
    else:
        st.warning("Demo mode.")

    if features_exist and FEATURE_LIST:
        st.success(f"Features loaded: {len(FEATURE_LIST)}")
    elif features_exist:
        st.success("Features file present.")
    else:
        st.info("Using default feature set.")

    st.markdown("---")
    model_info = {
        "Model Name": "new_model",
        "Model Type": "Stacking Ensemble",
        "Version": "20251004_211436",
        "Features": str(len(FEATURE_LIST)) if FEATURE_LIST else "20",
        "Classes": "2 (PLANET / NOT_PLANET)",
        "Training Samples": "23,289"
    }
    for k, v in model_info.items():
        st.metric(k, v)

    st.markdown("---")
    st.markdown("#### Data Requirements")
    st.markdown("""
    **Required Columns**
    - Time column (time, t, BJD, MJD, etc.)
    - Flux column (flux, f, brightness, etc.)

    **Recommended**
    - 100+ data points
    - Clean, continuous data
    """)

    st.markdown("---")
    show_features = st.checkbox("Show Extracted Features", value=False)
    show_visualization = st.checkbox("Show Visualizations", value=True)

# ================== Uploader ==================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader(
        "Upload Light Curve Data (CSV)",
        type=['csv'],
        help="CSV with time and flux. Column names auto-detected; manual selection available."
    )

# ================== Main ==================
if uploaded_file is not None:
    try:
        with st.spinner('Loading light curve data...'):
            df = pd.read_csv(uploaded_file)
            time.sleep(0.2)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(f"Detected columns: {', '.join(df.columns.tolist())}")
        st.markdown('</div>', unsafe_allow_html=True)

        time_col, flux_col = None, None
        for col in df.columns:
            cl = col.lower().strip()
            if cl in ['time','times','t','bjd','jd','mjd','timestamp']:
                time_col = col if time_col is None else time_col
            if cl in ['flux','fluxes','f','brightness','magnitude','mag','intensity']:
                flux_col = col if flux_col is None else flux_col

        if time_col is None or flux_col is None:
            st.warning("Auto-detection failed. Please select columns.")
            c1, c2 = st.columns(2)
            with c1:
                time_col = st.selectbox("TIME column", df.columns.tolist(), index=0 if len(df.columns)>0 else None)
            with c2:
                flux_col = st.selectbox("FLUX column", df.columns.tolist(), index=1 if len(df.columns)>1 else None)
        else:
            st.success(f"Auto-detected → Time: '{time_col}', Flux: '{flux_col}'")

        df_processed = df.copy()
        df_processed['time'] = df[time_col]
        df_processed['flux'] = df[flux_col]

        if time_col is None or flux_col is None:
            st.error("Please select both time and flux columns.")
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Data Points", f"{len(df_processed):,}")
            c2.metric("Time Span", f"{df_processed['time'].max() - df_processed['time'].min():.2f}")
            c3.metric("Flux Range", f"{df_processed['flux'].max() - df_processed['flux'].min():.6f}")
            st.dataframe(df_processed[['time','flux']].head(10), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("")
            c1, c2, c3 = st.columns([1,1,1])
            with c2:
                analyze_button = st.button("Run Exoplanet Detection", use_container_width=True)

            if analyze_button:
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("Step 1/5 — Preprocessing")
                progress_bar.progress(20); time.sleep(0.2)

                status_text.text("Step 2/5 — Extracting features")
                features = extract_features(df_processed)
                progress_bar.progress(40); time.sleep(0.2)
                if features: st.info(f"Extracted {len(features)} features.")

                status_text.text("Step 3/5 — Loading model")
                model = load_model()
                progress_bar.progress(60); time.sleep(0.15)

                status_text.text("Step 4/5 — Classification")
                results = predict_with_model(features, model)
                progress_bar.progress(80); time.sleep(0.2)

                status_text.text("Step 5/5 — Report generation")
                progress_bar.progress(100); time.sleep(0.15)
                progress_bar.empty(); status_text.empty()

                # ---- Results ----
                st.markdown("### Detection Results")
                st.markdown("---")

                cc1, cc2, cc3 = st.columns([1,2,1])
                with cc2:
                    if results['prediction'] == 'PLANET':
                        st.markdown('<div class="status positive">PLANET</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="status negative">NOT PLANET</div>', unsafe_allow_html=True)

                st.markdown("")
                st.markdown("#### Classification Confidence")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Overall Confidence</div><div class="metric-value">{results["confidence"]:.1%}</div></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Planet Probability</div><div class="metric-value">{results["planet_probability"]:.1%}</div></div>', unsafe_allow_html=True)
                with m3:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Not Planet Probability</div><div class="metric-value">{results["not_planet_probability"]:.1%}</div></div>', unsafe_allow_html=True)
                with m4:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Features Used</div><div class="metric-value">{results["feature_count"]}</div></div>', unsafe_allow_html=True)

                st.markdown("")
                i1, i2, i3 = st.columns(3)
                i1.metric("Model Type", results['model_type'])
                i2.metric("Model Version", results['model_version'])
                i3.metric("Training Dataset", "23,289 samples")

                # Guidance
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

                # Data quality indicators
                mean_f = features.get('mean_flux', float(df_processed['flux'].mean()))
                std_f = features.get('std_flux', float(df_processed['flux'].std()))
                cv_f = features.get('cv_flux', (std_f/mean_f) if mean_f else 0.0)
                st.markdown("#### Data Quality Indicators")
                st.markdown(f"- Signal-to-Noise (mean/std): **{(mean_f/std_f if std_f else 0):.2f}**")
                st.markdown(f"- Coefficient of Variation: **{cv_f:.4f}**")
                st.markdown(f"- Data Points: **{len(df_processed):,}**")

                # Features
                if show_features and features:
                    st.markdown("#### Extracted Features")
                    left, right = st.columns(2)
                    items = list(features.items())
                    mid = len(items) // 2
                    with left:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown("**Set A**")
                        for k, v in items[:mid]:
                            st.metric(k.replace('_',' ').title(), f"{v:.6f}" if isinstance(v,(int,float)) else str(v))
                        st.markdown('</div>', unsafe_allow_html=True)
                    with right:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown("**Set B**")
                        for k, v in items[mid:]:
                            st.metric(k.replace('_',' ').title(), f"{v:.6f}" if isinstance(v,(int,float)) else str(v))
                        st.markdown('</div>', unsafe_allow_html=True)

                # Visualization
                if show_visualization:
                    st.markdown("#### Light Curve Analysis")
                    fig = create_visualization_plot(df_processed, features)
                    st.plotly_chart(fig, use_container_width=True)

                # Download
                st.markdown("")
                cdl1, cdl2, cdl3 = st.columns([1,1,1])
                with cdl2:
                    full_report = {**results, **features}
                    report_df = pd.DataFrame([full_report])
                    csv = report_df.to_csv(index=False)
                    st.download_button(
                        label="Download Full Report (CSV)",
                        data=csv,
                        file_name=f"exoplanet_detection_report_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)

else:
    st.markdown("")
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        # 帶虛線邊框的上傳說明區（標題為純白）
        st.markdown("""
        <div class="upload-card">
            <h4>Upload Light Curve Data</h4>
            <p>Provide a CSV with time and flux columns. The system will auto-detect column names or allow manual selection.</p>
            <small>Supported time columns: TIME / T / BJD / JD / MJD · Supported flux columns: FLUX / F / BRIGHTNESS / MAG</small>
        </div>
        """, unsafe_allow_html=True)

# ================== Footer ==================
st.markdown("""
<hr style="border-color: rgba(36,49,74,.6); margin-top:2rem;"/>
<div style='text-align:center; color:#c6d2e6; padding-top: .6rem;'>
    <div style="font-size:.95rem;">Exoplanet Detection AI Platform — Stacking Ensemble v20251004_211436</div>
    <div style="font-size:.85rem; opacity:.85; margin-top:.2rem;">Training sources: TESS, Kepler, Confirmed Planets</div>
</div>
""", unsafe_allow_html=True)
