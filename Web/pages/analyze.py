import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from components.banner import render_banner
import time
import json
import os

# ========== Page Config ==========
st.set_page_config(
    page_title="Exoplanet Detection AI Pro",
    page_icon="Web/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)
render_banner()

# ---- Session State (persist across reruns) ----
if 'analysis_ready' not in st.session_state:
    st.session_state.analysis_ready = False     # 是否已有分析結果
if 'analysis' not in st.session_state:
    st.session_state.analysis = None            # {'features':..., 'extra':..., 'results':...}
if 'pp_range' not in st.session_state:
    st.session_state.pp_range = (0.50, 1.00)    # Planet Prob 篩選區間（分析前可設定）
if 'last_upload_token' not in st.session_state:
    st.session_state.last_upload_token = None   # 用來偵測是否換了新檔

# ---- Hide default header/footer ----
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ========== Paths ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
FEATURE_LIST_PATH = os.path.join(MODELS_DIR, 'feature_list.json')
MODEL_PATH = os.path.join(MODELS_DIR, '\exoplanet_model_20251005_231130.pkl')

# ========== THEME ==========
def apply_premium_theme():
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1729 100%); }
        .main-title{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
            font-size: 3.5rem; font-weight: 800; text-align: center; margin-bottom: .5rem; letter-spacing: 1px;
        }
        .subtitle{ text-align:center; color:#fff; font-size:1.4rem; margin-bottom:2rem; font-weight:500; }

        .analysis-card{
            background: linear-gradient(135deg, rgba(22,33,62,.7), rgba(15,52,96,.7));
            border:2px solid rgba(102,126,234,.3); border-radius:20px; padding:2rem; margin:1.5rem 0;
            box-shadow:0 4px 16px rgba(31,38,135,.3); backdrop-filter: blur(10px); transition:.3s;
        }
        .analysis-card:hover{ transform: translateY(-3px); background: linear-gradient(135deg, rgba(22,33,62,.95), rgba(15,52,96,.95)); border-color: rgba(102,126,234,.6); box-shadow:0 12px 40px rgba(102,126,234,.6); }
        .analysis-card h3,.analysis-card h4{ color:#fff !important; font-weight:700; font-size:1.4rem; }
        .analysis-card p{ color:#d0d0d0 !important; font-size:1.1rem !important; }

        .metric-card{
            background: linear-gradient(135deg, rgba(102,126,234,.15), rgba(118,75,162,.15));
            border:2px solid rgba(102,126,234,.4); border-radius:15px; padding:2rem 1.5rem; text-align:center; transition:.3s;
        }
        .metric-value{ font-size:2.8rem; font-weight:700; color:#4facfe; margin:.8rem 0; }
        .metric-label{ color:#d0d0d0; font-size:1.15rem; text-transform:uppercase; letter-spacing:1px; font-weight:700; }

        .stButton > button{
            background: linear-gradient(135deg, rgba(102,126,234,.8) 0%, rgba(118,75,162,.8) 100%) !important;
            color:#fff !important; border:2px solid rgba(102,126,234,.5) !important; border-radius:15px !important;
            padding:1.2rem 3rem !important; font-size:1.2rem !important; font-weight:600 !important;
            box-shadow:0 4px 15px rgba(102,126,234,.3) !important; transition:.3s !important; width:100%;
        }

        /* Results-scoped styles */
        .rs-section { margin: 0.5rem 0 2rem; }
        .rs-title { font-size: 1.6rem; font-weight: 800; margin: .25rem 0 .75rem; color: #fff; }
        .rs-badge {
          display:flex; align-items:center; justify-content:center;
          padding: 1.1rem 1.6rem; border-radius: 18px; font-weight: 900; font-size: 1.2rem;
          border: 2px solid rgba(255,255,255,.25); text-transform: uppercase; letter-spacing: .5px;
        }
        .rs-badge.ok  { background: linear-gradient(135deg,#48bb78 0%,#38a169 100%); border-color:#68d391; box-shadow:0 8px 24px rgba(72,187,120,.35); }
        .rs-badge.no  { background: linear-gradient(135deg,#fc8181 0%,#f56565 100%); border-color:#feb2b2; box-shadow:0 8px 24px rgba(245,101,101,.35); }
        .rs-grid {
          display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          gap: 14px; margin: .5rem 0 1.25rem;
        }
        .rs-card {
          background: linear-gradient(135deg, rgba(102,126,234,.15), rgba(118,75,162,.15));
          border: 1px solid rgba(102,126,234,.35);
          border-radius: 14px; padding: 14px 16px; text-align: center;
          box-shadow: 0 4px 14px rgba(16,24,48,.35);
        }
        .rs-card .lbl { color:#d7dbee; font-size: .95rem; font-weight: 700; letter-spacing:.4px; text-transform: uppercase; }
        .rs-card .val { color:#4facfe; font-size: 1.9rem; font-weight: 800; margin-top: .25rem; }
        .rs-divider { height:1px; background: linear-gradient(90deg, transparent, rgba(102,126,234,.5), transparent); margin: 1.25rem 0; border:0; }

        /* Uploader readability */
        .stFileUploader{ border:3px dashed rgba(102,126,234,.3) !important; border-radius:20px !important; padding:2rem !important; background: rgba(255,255,255,.92) !important; }
        .stFileUploader, .stFileUploader *{ color:#111 !important; }
    </style>
    """, unsafe_allow_html=True)

apply_premium_theme()

# ========== Feature list ==========
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
        return None
    except Exception as e:
        st.error(f"Error loading feature list: {str(e)}")
        return None

FEATURE_LIST = load_feature_list()

# ========== Feature extraction ==========
def extract_features(df):
    try:
        features = {}
        flux = df['flux'].values
        time_vals = df['time'].values
        all_features = {}
        if len(flux) < 10:
            st.warning("Very few data points detected. Results may be unreliable.")

        all_features['mean_flux'] = np.mean(flux)
        all_features['std_flux'] = np.std(flux)
        all_features['min_flux'] = np.min(flux)
        all_features['max_flux'] = np.max(flux)
        all_features['median_flux'] = np.median(flux)
        all_features['flux_range'] = np.max(flux) - np.min(flux)
        all_features['cv_flux'] = np.std(flux) / np.mean(flux) if np.mean(flux) != 0 else 0

        all_features['skewness'] = stats.skew(flux, bias=False) if len(flux) > 2 else 0.0
        all_features['kurtosis'] = stats.kurtosis(flux, bias=False) if len(flux) > 3 else 0.0
        all_features['percentile_75'] = np.percentile(flux, 75)
        all_features['percentile_25'] = np.percentile(flux, 25)
        all_features['percentile_90'] = np.percentile(flux, 90)

        if np.max(time_vals) != np.min(time_vals):
            tn = (time_vals - np.min(time_vals)) / (np.max(time_vals) - np.min(time_vals))
        else:
            tn = time_vals

        if len(flux) >= 2:
            slope, intercept, r_value, _, _ = stats.linregress(tn, flux)
        else:
            slope, intercept, r_value = 0.0, float(np.mean(flux)), 0.0

        all_features['trend_slope'] = slope
        all_features['trend_r2'] = r_value**2
        all_features['linear_fit_error'] = float(np.mean(np.abs(flux - (slope * tn + intercept)))) if len(flux) else 0.0

        flux_diff = np.diff(flux) if len(flux) >= 2 else np.array([0.0])
        all_features['mean_absolute_deviation'] = float(np.mean(np.abs(flux - np.mean(flux)))) if len(flux) else 0.0
        all_features['flux_diff_mean'] = float(np.mean(flux_diff))
        all_features['flux_diff_std'] = float(np.std(flux_diff))
        all_features['flux_diff_max'] = float(np.max(np.abs(flux_diff)))

        window = max(5, len(flux)//20) if len(flux) >= 5 else 5
        s = pd.Series(flux)
        rolling_std = s.rolling(window=window, center=True).std()
        all_features['rolling_std_mean'] = float(np.nanmean(rolling_std))
        all_features['rolling_std_max'] = float(np.nanmax(rolling_std))
        all_features['rolling_std_min'] = float(np.nanmin(rolling_std))

        all_features['data_points'] = int(len(flux))
        all_features['time_span'] = float(np.max(time_vals) - np.min(time_vals)) if len(time_vals) else 0.0

        all_features['snr'] = float(all_features['mean_flux'] / all_features['std_flux']) if all_features['std_flux'] != 0 else 0.0
        all_features['data_quality_score'] = float(min(100, (len(flux)/100)*50 + all_features['snr']*10))

        if FEATURE_LIST:
            for f in FEATURE_LIST:
                features[f] = all_features.get(f, 0)
        else:
            for f in ['mean_flux','std_flux','min_flux','max_flux','median_flux','flux_range','cv_flux','skewness','kurtosis','percentile_75','trend_slope','trend_r2','linear_fit_error','mean_absolute_deviation','flux_diff_mean','flux_diff_std','flux_diff_max','rolling_std_mean','rolling_std_max','rolling_std_min']:
                features[f] = all_features.get(f, 0)

        return features, all_features
    except Exception as e:
        st.error(f"Feature extraction failed: {str(e)}")
        return {}, {}

# ========== Model loading ==========
def load_model():
    try:
        import joblib
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            
            # 嘗試修復模型的隨機性（深入修復）
            try:
                # 如果是 StackingClassifier
                if hasattr(model, 'estimators_'):
                    for name, estimator in model.estimators_:
                        if hasattr(estimator, 'random_state'):
                            if estimator.random_state is None:
                                estimator.random_state = 42
                                st.sidebar.warning(f"⚠️ Fixed random_state for {name}")
                
                # 修復 final_estimator
                if hasattr(model, 'final_estimator_'):
                    if hasattr(model.final_estimator_, 'random_state'):
                        if model.final_estimator_.random_state is None:
                            model.final_estimator_.random_state = 42
                            st.sidebar.warning("⚠️ Fixed random_state for final_estimator")
                
                # 修復單一模型
                if hasattr(model, 'random_state') and model.random_state is None:
                    model.random_state = 42
                    st.sidebar.warning("⚠️ Fixed random_state for model")
                    
            except Exception as fix_error:
                st.sidebar.error(f"Could not fix model randomness: {str(fix_error)}")
            
            return model
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# ========== Predict ==========
def predict_with_model(features_dict, model=None):
    if not features_dict:
        st.warning("⚠️ No features extracted - using fallback prediction")
        return {'prediction':'NOT_PLANET','confidence':0.5,'planet_probability':0.3,'not_planet_probability':0.7,'model_version':'20251004_211436','model_type':'Stacking Ensemble','feature_count':0}
    
    X = pd.DataFrame([features_dict])
    
    if model is not None:
        try:
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            not_p, p = float(proba[0]), float(proba[1])
            conf = float(p if pred=='PLANET' else not_p)
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.warning("⚠️ Using random fallback values - model may have feature mismatch")
            pred = np.random.choice(['PLANET','NOT_PLANET'], p=[0.35,0.65])
            p = float(np.random.uniform(0.3,0.7)); not_p = 1-p; conf=max(p,not_p)
    else:
        st.warning("⚠️ Model not loaded - using random fallback values (results will vary each run)")
        pred = np.random.choice(['PLANET','NOT_PLANET'], p=[0.35,0.65])
        p = float(np.random.uniform(0.3,0.7)); not_p = 1-p; conf=max(p,not_p)
    
    return {'prediction':str(pred),'confidence':float(conf),'planet_probability':float(p),'not_planet_probability':float(not_p),'model_version':'20251004_211436','model_type':'Stacking Ensemble','feature_count':len(features_dict)}

# ========== Plot ==========
def create_visualization_plot(df, features, extra_features):
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Light Curve Time Series','Flux Distribution & Statistics','Detrended Signal Analysis','Feature Importance Ranking','Rolling Statistics','Signal Quality Metrics'),
        specs=[[{"secondary_y": False},{"secondary_y": False}],
               [{"secondary_y": False},{"type":"bar"}],
               [{"secondary_y": True},{"type":"indicator"}]],
        vertical_spacing=0.10, horizontal_spacing=0.12
    )

    mean_flux = features.get('mean_flux', df['flux'].mean())
    std_flux = features.get('std_flux', df['flux'].std())

    fig.add_trace(go.Scatter(x=df['time'], y=df['flux'], mode='lines+markers', name='Light Curve',
                             line=dict(color='#00f5ff', width=2.5),
                             marker=dict(size=4, color='#667eea', opacity=0.6),
                             hovertemplate='<b>Time:</b> %{x:.4f}<br><b>Flux:</b> %{y:.8f}<extra></extra>'),
                  row=1, col=1)

    fig.add_hline(y=mean_flux, line_dash="dash", line_color="#00ff88", line_width=3,
                  annotation_text="Mean Flux", annotation_font_color="#00ff88", annotation_font_size=12, row=1, col=1)
    fig.add_hline(y=mean_flux+std_flux, line_dash="dot", line_color="#ff0080", line_width=2, annotation_text="+1σ", row=1, col=1)
    fig.add_hline(y=mean_flux-std_flux, line_dash="dot", line_color="#ff0080", line_width=2, annotation_text="-1σ", row=1, col=1)

    fig.add_trace(go.Histogram(x=df['flux'], name='Distribution', nbinsx=40,
                               marker=dict(color='#667eea', opacity=0.7, line=dict(color='#00f5ff', width=1))),
                  row=1, col=2)

    flux = df['flux'].values
    tn = (df['time']-df['time'].min())/(df['time'].max()-df['time'].min()) if (df['time'].max()!=df['time'].min()) else df['time']
    if len(flux) >= 2:
        slope, intercept, _, _, _ = stats.linregress(tn, flux)
    else:
        slope, intercept = 0.0, float(np.mean(flux)) if len(flux) else 0.0
    detrended = flux - (slope*tn + intercept)
    fig.add_trace(go.Scatter(x=df['time'], y=detrended, mode='lines', name='Detrended Signal',
                             line=dict(color='#f093fb', width=2.5)),
                  row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#00f5ff", line_width=2, row=2, col=1)

    feature_names = ['Mean','Std Dev','Range','CV','Skewness','Kurtosis','Trend','SNR']
    feature_values = [abs(features.get('mean_flux',mean_flux))/10000,
                      abs(features.get('std_flux',std_flux))/1000,
                      abs(features.get('flux_range',0))/1000,
                      abs(features.get('cv_flux',0))*1000,
                      abs(features.get('skewness',0))*100,
                      abs(features.get('kurtosis',0))*10,
                      abs(features.get('trend_slope',0))*10000,
                      abs(extra_features.get('snr',0))]
    colors = ['#00f5ff','#667eea','#764ba2','#f093fb','#00ff88','#ffbb00','#ff0080','#00d4aa']
    fig.add_trace(go.Bar(x=feature_names, y=feature_values, name='Feature Values',
                         marker=dict(color=colors, opacity=.8, line=dict(color='#fff', width=2))),
                  row=2, col=2)

    window = max(5, len(flux)//20) if len(flux) >= 5 else 5
    rolling_mean = pd.Series(flux).rolling(window=window, center=True).mean()
    rolling_std = pd.Series(flux).rolling(window=window, center=True).std()
    fig.add_trace(go.Scatter(x=df['time'], y=rolling_mean, mode='lines', name='Rolling Mean', line=dict(color='#00ff88', width=3)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=rolling_std, mode='lines', name='Rolling Std', line=dict(color='#ff0080', width=3, dash='dash')), row=3, col=1, secondary_y=True)

    quality_score = extra_features.get('data_quality_score', 50)
    fig.add_trace(go.Indicator(mode="gauge+number+delta", value=quality_score,
                               title={'text':"Data Quality",'font':{'size':20,'color':'#00f5ff'}},
                               delta={'reference':75,'increasing':{'color':"#00ff88"}},
                               gauge={'axis':{'range':[None,100],'tickcolor':'#00f5ff'},
                                      'bar':{'color':"#00f5ff"}, 'bgcolor':"rgba(0,0,0,.5)",
                                      'borderwidth':2,'bordercolor':"#667eea",
                                      'steps':[{'range':[0,50],'color':'rgba(255,0,128,.3)'},
                                               {'range':[50,75],'color':'rgba(255,187,0,.3)'},
                                               {'range':[75,100],'color':'rgba(0,255,136,.3)'}],
                                      'threshold':{'line':{'color':"#f093fb",'width':4},'thickness':.75,'value':90}}),
                  row=3, col=2)

    fig.update_layout(height=1200, showlegend=True, paper_bgcolor='rgba(10,14,39,.95)', plot_bgcolor='rgba(15,25,50,.9)',
                      font=dict(color='#e0f0ff', size=13, family='Rajdhani'))
    fig.update_xaxes(gridcolor='rgba(0,245,255,.2)', gridwidth=1, showline=True, linewidth=2, linecolor='#00f5ff')
    fig.update_yaxes(gridcolor='rgba(102,126,234,.2)', gridwidth=1, showline=True, linewidth=2, linecolor='#667eea')
    for a in fig['layout']['annotations']:
        a['font'] = dict(size=16, color='#00f5ff', family='Orbitron', weight='bold')
    return fig

# ========== App Header ==========
st.markdown('<h1 class="main-title">Exoplanet Detection AI Platform</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Machine Learning · Stacking Ensemble · Real-time Analysis</p>', unsafe_allow_html=True)

# ========== Sidebar ==========
with st.sidebar:
    st.markdown("### System Status")
    model_exists = os.path.exists(MODEL_PATH); features_exist = os.path.exists(FEATURE_LIST_PATH)
    c1,c2 = st.columns(2)
    with c1: st.success("Model available") if model_exists else st.warning("Demo mode")
    with c2: st.success("Features loaded") if features_exist else st.info("Using defaults")
    st.markdown("---")
    st.markdown("### Model Info")
    specs = {"Name":"Ensemble","Version":"v20251004","Features": str(len(FEATURE_LIST)) if FEATURE_LIST else "20","Classes":"2","Samples":"23,289"}
    for k,v in specs.items(): st.metric(k, v)
    st.markdown("---")
    st.markdown("### Settings")
    show_features = st.checkbox("Show Features", value=False)
    show_visualization = st.checkbox("Show Visualizations", value=True)
    show_advanced = st.checkbox("Advanced Analysis", value=False)
    auto_analyze = st.checkbox("Auto-Analyze", value=False)
    st.markdown("---")
    st.markdown("### Requirements")
    st.info("Required Columns:\n• Time series data\n• Flux measurements\n\nOptimal:\n• 100+ data points\n• Clean, continuous data\n• Normalized flux")
    st.markdown("---")
    st.markdown("### Workflow")
    st.markdown("1. Upload CSV data  \n2. Set probability range  \n3. Run analysis  \n4. View results  \n5. Export report")
    st.markdown("---")
    with st.expander("Debug Info", expanded=False):
        st.code(f"Model exists: {os.path.exists(MODEL_PATH)}")
        st.code(f"Features exists: {os.path.exists(FEATURE_LIST_PATH)}")
        
        # 模型類型檢測
        if os.path.exists(MODEL_PATH):
            model = load_model()
            if model is not None:
                st.code(f"Model type: {type(model).__name__}")
                st.code(f"Model class: {model.__class__.__module__}.{model.__class__.__name__}")
        
        if st.session_state.analysis_ready:
            st.code(f"Last analysis: {st.session_state.analysis.get('timestamp', 'N/A')}")
            st.code(f"Feature hash: {st.session_state.analysis.get('feature_hash', 'N/A')}")
            st.code(f"Planet prob: {st.session_state.analysis['results']['planet_probability']:.6f}")
            
            st.markdown("---")
            st.markdown("**🧪 Diagnostics**")
            st.info("Run tests after uploading data and completing analysis")

st.markdown("<br>", unsafe_allow_html=True)

# ========== Uploader ==========
c1,c2,c3 = st.columns([1,3,1])
with c2:
    uploaded_file = st.file_uploader(
        "Upload Light Curve Data (CSV Format)",
        type=['csv'],
        help="CSV file with time and flux columns - auto-detection enabled"
    )

# ==============================================
# ============== Main Branch ===================
# ==============================================
if uploaded_file is not None:
    try:
        with st.spinner('Loading data...'):
            df = pd.read_csv(uploaded_file); time.sleep(0.2)

        # 以檔名 + 大小建立 token；若換檔就重置狀態（注意：這在建立 slider 之前執行）
        try:
            size_hint = uploaded_file.size
        except Exception:
            size_hint = len(uploaded_file.getvalue())
        upload_token = f"{uploaded_file.name}:{size_hint}"
        if st.session_state.last_upload_token != upload_token:
            st.session_state.last_upload_token = upload_token
            st.session_state.analysis_ready = False
            st.session_state.analysis = None
            st.session_state.pp_range = (0.50, 1.00)  # OK：此時還沒畫出 slider

        st.success(f"Data loaded: {len(df):,} records")
        st.info(f"Detected columns: {', '.join(df.columns.tolist())}")

        # Column detection
        time_col = None; flux_col = None
        for col in df.columns:
            cl = col.lower().strip()
            if cl in ['time','times','t','bjd','jd','mjd','timestamp','date']: time_col = col
            if cl in ['flux','fluxes','f','brightness','magnitude','mag','intensity','signal']: flux_col = col

        if time_col is None or flux_col is None:
            st.warning("Please select columns manually:")
            cc1,cc2 = st.columns(2)
            with cc1: time_col = st.selectbox("Select TIME column:", df.columns.tolist(), index=0)
            with cc2: flux_col = st.selectbox("Select FLUX column:", df.columns.tolist(), index=1 if len(df.columns)>1 else 0)
        else:
            st.success(f"Auto-detected: Time = '{time_col}', Flux = '{flux_col}'")

        # Normalize names
        df_processed = df.copy()
        df_processed['time'] = df[time_col]
        df_processed['flux'] = df[flux_col]

        # ---------- Data Preview ----------
        with st.expander("Data Preview & Statistics", expanded=True):
            m1,m2,m3,m4 = st.columns(4)
            with m1: st.markdown(f"<div class='metric-card'><div class='metric-label'>Data Points</div><div class='metric-value'>{len(df_processed):,}</div></div>", unsafe_allow_html=True)
            with m2: st.markdown(f"<div class='metric-card'><div class='metric-label'>Time Span</div><div class='metric-value'>{df_processed['time'].max() - df_processed['time'].min():.2f}</div></div>", unsafe_allow_html=True)
            with m3: st.markdown(f"<div class='metric-card'><div class='metric-label'>Mean Flux</div><div class='metric-value'>{df_processed['flux'].mean():.4f}</div></div>", unsafe_allow_html=True)
            with m4: st.markdown(f"<div class='metric-card'><div class='metric-label'>Std Dev</div><div class='metric-value'>{df_processed['flux'].std():.4f}</div></div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(df_processed[['time','flux']].head(15), use_container_width=True, height=300)

        # ---------- 🧪 Diagnostic Tests ----------
        if st.session_state.analysis_ready:
            with st.expander("🧪 Model Diagnostics", expanded=False):
                st.markdown("### Test Model Consistency")
                st.info("These tests help identify if your model produces consistent results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔬 Test Feature Extraction", help="Extract features 5 times to check consistency", use_container_width=True):
                        with st.spinner("Testing feature extraction..."):
                            import hashlib
                            hashes = []
                            for i in range(5):
                                features, _ = extract_features(df_processed)
                                feature_hash = hashlib.md5(str(sorted(features.items())).encode()).hexdigest()[:8]
                                hashes.append(feature_hash)
                                st.code(f"Extraction {i+1}: {feature_hash}")
                            
                            if len(set(hashes)) == 1:
                                st.success("✅ Feature extraction is DETERMINISTIC")
                            else:
                                st.error("❌ Feature extraction is NON-DETERMINISTIC!")
                                st.warning("This shouldn't happen - check for random operations in extract_features()")
                
                with col2:
                    if st.button("🧪 Test Model Predictions", help="Run 5 predictions to check if results are consistent", use_container_width=True):
                        with st.spinner("Testing model predictions..."):
                            features = st.session_state.analysis['features']
                            model = load_model()
                            probs = []
                            for i in range(5):
                                result = predict_with_model(features, model)
                                probs.append(result['planet_probability'])
                                st.code(f"Prediction {i+1}: {result['planet_probability']:.8f}")
                            
                            if len(set(probs)) == 1:
                                st.success("✅ Model is DETERMINISTIC - all predictions identical!")
                            else:
                                st.error(f"❌ Model is NON-DETERMINISTIC - got {len(set(probs))} different values!")
                                st.warning("**Possible causes:**")
                                st.code("1. Model uses dropout/random sampling during inference")
                                st.code("2. Model trained without fixed random_state")
                                st.code("3. Ensemble uses random tie-breaking")
                                st.info("💡 **Solution:** Retrain model with:\n- random_state=42 for all estimators\n- model.set_params(random_state=42)")
                                
                                # 顯示變異程度
                                import numpy as np
                                std = np.std(probs)
                                range_val = max(probs) - min(probs)
                                st.metric("Standard Deviation", f"{std:.6f}")
                                st.metric("Range", f"{range_val:.6f} ({range_val*100:.2f}%)")

        st.markdown("<br>", unsafe_allow_html=True)

        # ---------- (Pre-analysis) Planet Prob Preference ----------
        pref1, pref2, pref3 = st.columns([1,3,1])
        with pref2:
            st.markdown('<h3 style="color:#ffffff; margin-bottom:8px;">Custom Detection Threshold</h3>', unsafe_allow_html=True)
            st.markdown('<div style="color:#e8eef8; margin-bottom:6px;">Set your probability range - signals within this range will be classified as PLANET:</div>', unsafe_allow_html=True)
            if st.session_state.analysis_ready:
                st.markdown('<div style="color:#ffd700; font-size:0.9rem; margin-bottom:6px;">💡 Tip: Adjusting the slider below will update the detection result instantly without re-running the analysis.</div>', unsafe_allow_html=True)
            pp_range = st.slider(
                "",  
                min_value=0.0, max_value=1.0,
                value=st.session_state.pp_range, step=0.01,
                key="pp_range",
                help="Define your custom threshold range. Default model uses 50%, but you can adjust this based on your requirements (e.g., [0.70-1.00] for high confidence, [0.30-0.70] for uncertain cases)"
            )

        # ---------- Analyze Button ----------
        a1,a2,a3 = st.columns([1,2,1])
        with a2:
            run_clicked = st.button("Run Analysis", use_container_width=True, type="primary")

        should_run_now = run_clicked or (auto_analyze and not st.session_state.analysis_ready)

        # 執行分析並存入 session（之後拉動滑桿介面不會消失）
        if should_run_now:
            progress = st.progress(0); status = st.empty()
            status.markdown("### Step 1/5: Preprocessing data..."); progress.progress(10); time.sleep(0.3)
            status.markdown("### Step 2/5: Extracting features..."); progress.progress(30)
            features, extra = extract_features(df_processed)
            status.markdown("### Step 3/5: Loading model..."); progress.progress(50)
            model = load_model()
            if model is not None:
                status.markdown("### ✅ Model loaded successfully"); time.sleep(0.3)
            else:
                status.markdown("### ⚠️ Model not found - using demo mode"); time.sleep(0.3)
            status.markdown("### Step 4/5: Running classification..."); progress.progress(75)
            results = predict_with_model(features, model); time.sleep(0.2)
            status.markdown("### Step 5/5: Generating results..."); progress.progress(100); time.sleep(0.1)
            status.empty(); progress.empty()

            # 儲存分析結果和時間戳記用於調試
            import hashlib
            feature_hash = hashlib.md5(str(sorted(features.items())).encode()).hexdigest()[:8]
            st.session_state.analysis = {
                'features': features, 
                'extra': extra, 
                'results': results,
                'timestamp': time.strftime('%H:%M:%S'),
                'feature_hash': feature_hash
            }
            st.session_state.analysis_ready = True
            
            # 顯示分析完成訊息
            st.success(f"✅ Analysis completed at {st.session_state.analysis['timestamp']} | Feature hash: {feature_hash}")

        # ---------- 顯示分析結果（從 session 讀取；滑桿變動也會保留） ----------
        if st.session_state.analysis_ready and st.session_state.analysis is not None:
            features = st.session_state.analysis['features']
            extra    = st.session_state.analysis['extra']
            results  = st.session_state.analysis['results']

            st.markdown('<div class="rs-divider"></div>', unsafe_allow_html=True)
            
            # 顯示分析時間戳記
            analysis_time = st.session_state.analysis.get('timestamp', 'Unknown')
            feature_hash = st.session_state.analysis.get('feature_hash', 'N/A')
            st.info(f"📊 Showing results from analysis at **{analysis_time}** (Hash: {feature_hash})")
            
            st.markdown('<div class="rs-section rs-title">Detection Results</div>', unsafe_allow_html=True)

            # ---- 根據使用者設定的範圍來決定最終結果 ----
            p = float(results['planet_probability'])
            # 直接使用 session_state 中的值，因為 slider 的 key 會自動更新它
            current_range = st.session_state.pp_range
            in_range = (current_range[0] <= p <= current_range[1])
            
            # 最終判定：只看機率是否在使用者設定的範圍內
            # 使用者設定範圍就是在自訂閾值，所以不用再考慮模型原始預測
            is_planet_detected = in_range
            badge_cls = "ok" if is_planet_detected else "no"
            label = "PLANET DETECTED" if is_planet_detected else "NO PLANET DETECTED"
            st.markdown(f'<div class="rs-badge {badge_cls}">{label}</div>', unsafe_allow_html=True)

            st.markdown('<div class="rs-section rs-title">Classification Metrics</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="rs-grid">
              <div class="rs-card"><div class="lbl">Planet Prob.</div><div class="val">{results['planet_probability']:.1%}</div></div>
              <div class="rs-card"><div class="lbl">Your Threshold</div><div class="val">{current_range[0]:.0%}–{current_range[1]:.0%}</div></div>
              <div class="rs-card"><div class="lbl">Feature Count</div><div class="val">{results['feature_count']}</div></div>
            </div>
            """, unsafe_allow_html=True)

            # ---- 顯示判定說明 ----
            if in_range:
                st.success(f"✅ Planet probability ({p:.1%}) is within your specified range ({current_range[0]:.0%}–{current_range[1]:.0%})")
            else:
                st.warning(f"⚠️ Planet probability ({p:.1%}) is outside your specified range ({current_range[0]:.0%}–{current_range[1]:.0%})")

            # --- Model / Run Info ---
            st.markdown(f"""
            <div class="rs-grid">
              <div class="rs-card"><div class="lbl">Model Type</div><div class="val">{results['model_type']}</div></div>
              <div class="rs-card"><div class="lbl">Version</div><div class="val">{results['model_version']}</div></div>
              <div class="rs-card"><div class="lbl">Dataset</div><div class="val">23,289</div></div>
            </div>
            """, unsafe_allow_html=True)

            # --- Features (optional) ---
            if show_features and features:
                st.markdown('<div class="rs-section rs-title">Extracted Features</div>', unsafe_allow_html=True)
                left, right = st.columns(2)
                items = list(features.items()); mid = len(items) // 2
                with left:
                    st.markdown("<div class='analysis-card'><h3>Primary</h3>", unsafe_allow_html=True)
                    for k, v in items[:mid]:
                        st.markdown(f"<div class='rs-card'><div class='lbl'>{k.replace('_',' ').title()}</div><div class='val'>{v:.6f}</div></div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                with right:
                    st.markdown("<div class='analysis-card'><h3>Secondary</h3>", unsafe_allow_html=True)
                    for k, v in items[mid:]:
                        st.markdown(f"<div class='rs-card'><div class='lbl'>{k.replace('_',' ').title()}</div><div class='val'>{v:.6f}</div></div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            # --- Visualization (optional) ---
            if show_visualization:
                st.markdown('<div class="rs-section rs-title">Light Curve Analysis</div>', unsafe_allow_html=True)
                fig = create_visualization_plot(df_processed, features, extra)
                st.plotly_chart(fig, use_container_width=True)

            # --- Advanced (optional) ---
            if show_advanced:
                st.markdown('<div class="rs-section rs-title">Advanced Analysis</div>', unsafe_allow_html=True)
                cA, cB = st.columns(2)
                with cA:
                    st.markdown("<div class='analysis-card'><h3>Frequency Domain</h3>", unsafe_allow_html=True)
                    from scipy.fft import fft, fftfreq
                    F = fft(df_processed['flux'].values); n = len(df_processed)
                    dt = float(df_processed['time'].diff().mean()) if df_processed['time'].diff().notna().any() else 1.0
                    dt = 1.0 if dt == 0 else dt
                    freq = fftfreq(n, dt)
                    dom_idx = np.argmax(np.abs(F[1:n//2])) + 1 if n > 2 else 0
                    dom_f = freq[dom_idx] if dom_idx < len(freq) else 0.0
                    peak = float(np.max(np.abs(F[1:n//2]))) if n > 2 else 0.0
                    st.markdown(f"<div class='rs-card'><div class='lbl'>Dominant Frequency</div><div class='val'>{dom_f:.6f}</div></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='rs-card'><div class='lbl'>FFT Peak Power</div><div class='val'>{peak:.2f}</div></div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                with cB:
                    st.markdown("<div class='analysis-card'><h3>Statistical Tests</h3>", unsafe_allow_html=True)
                    from scipy.stats import normaltest
                    try:
                        if len(df_processed['flux']) >= 8:
                            _, pval = normaltest(df_processed['flux'])
                        else:
                            pval = np.nan
                    except Exception:
                        pval = np.nan
                    st.markdown(f"<div class='rs-card'><div class='lbl'>Normality p-value</div><div class='val'>{(pval if not np.isnan(pval) else 0):.6f}</div></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='rs-card'><div class='lbl'>Skewness</div><div class='val'>{extra.get('skewness',0):.4f}</div></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='rs-card'><div class='lbl'>Kurtosis</div><div class='val'>{extra.get('kurtosis',0):.4f}</div></div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            # --- Export ---
            full = {**results, **features, **extra,
                    "pp_range_min": current_range[0],
                    "pp_range_max": current_range[1],
                    "pp_in_range": in_range,
                    "final_detection": "PLANET" if is_planet_detected else "NO_PLANET",
                    "model_raw_prediction": results['prediction']}
            csv = pd.DataFrame([full]).to_csv(index=False)
            ts = time.strftime('%Y%m%d_%H%M%S')
            st.download_button(
                "Download Complete Report (CSV)",
                data=csv,
                file_name=f"exoplanet_analysis_{ts}.csv",
                mime="text/csv",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Error: {str(e)}")
        with st.expander("Error Details", expanded=False):
            st.exception(e)

# ==============================================
# ============== Empty State ===================
# ==============================================
else:
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,3,1])
    with c2:
        st.markdown("""
        <div style='text-align: center; padding: 4rem; background: rgba(22,33,62,.7); border-radius:25px; border:3px dashed rgba(102,126,234,.6);'>
            <h2 style='color:#fff; margin-bottom:1.5rem; font-size:2.5rem; font-weight:700;'>Upload Light Curve Data</h2>
            <p style='color:#fff; font-size:1.35rem; margin-bottom:2rem; font-weight:500;'>CSV format with time and flux measurements</p>
            <div style='background: rgba(10,20,40,.6); padding:2rem; border-radius:15px; border:2px solid rgba(102,126,234,.3);'>
                <p style='color:#fff; font-size:1.2rem; margin:.9rem 0; font-weight:500;'>• CSV Format Supported</p>
                <p style='color:#fff; font-size:1.2rem; margin:.9rem 0; font-weight:500;'>• Auto Column Detection</p>
                <p style='color:#fff; font-size:1.2rem; margin:.9rem 0; font-weight:500;'>• Real-Time Processing</p>
                <p style='color:#fff; font-size:1.2rem; margin:.9rem 0; font-weight:500;'>• High Accuracy AI Model</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ========== Footer ==========
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#fff; padding:2.5rem; border-top:2px solid rgba(102,126,234,.4); background:rgba(10,20,40,.5);'>
    <p style='font-size:1.25rem; margin-bottom:1rem; color:#fff; font-weight:600;'>Exoplanet Detection AI Platform</p>
    <p style='font-size:1.1rem; color:#fff;'>Stacking Ensemble Model v20251004_211436 | Trained on 23,289 Samples</p>
    <p style='font-size:1rem; color:#e0e0e0; margin-top:1rem;'>TESS · Kepler · Confirmed Planets Datasets</p>
</div>
""", unsafe_allow_html=True)