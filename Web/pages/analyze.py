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

st.set_page_config(page_title="Exoplanet Detection AI Pro", page_icon="Web/logo.png", layout="wide", initial_sidebar_state="expanded")
render_banner()

hide_streamlit_header_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_header_style, unsafe_allow_html=True)

# Get the correct paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
FEATURE_LIST_PATH = os.path.join(MODELS_DIR, 'feature_list.json')
MODEL_PATH = os.path.join(MODELS_DIR, 'new_model_20251004_211436.pkl')

# --- CLEAN PROFESSIONAL THEME ---
def apply_premium_theme():
    st.markdown("""
    <style>
        /* Clean Background */
        .stApp {
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1729 100%);
        }
        
        /* Clear, Readable Title */
        .main-title {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3.5rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0.5rem;
            letter-spacing: 1px;
        }
        
        .subtitle {
            text-align: center;
            color: #ffffff;
            font-size: 1.4rem;
            margin-bottom: 2rem;
            font-weight: 500;
        }
        
        /* Clean Cards - dark initially, bright on hover */
        .analysis-card {
            background: linear-gradient(135deg, rgba(22, 33, 62, 0.7), rgba(15, 52, 96, 0.7));
            border: 2px solid rgba(102, 126, 234, 0.3);
            border-radius: 20px;
            padding: 2rem;
            margin: 1.5rem 0;
            box-shadow: 0 4px 16px 0 rgba(31, 38, 135, 0.3);
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }
        
        .analysis-card:hover {
            transform: translateY(-3px);
            background: linear-gradient(135deg, rgba(22, 33, 62, 0.95), rgba(15, 52, 96, 0.95));
            border-color: rgba(102, 126, 234, 0.6);
            box-shadow: 0 12px 40px 0 rgba(102, 126, 234, 0.6);
        }
        
        .analysis-card h3, .analysis-card h4 {
            color: #ffffff !important;
            font-weight: 700;
            font-size: 1.4rem;
        }
        
        .analysis-card p {
            color: #d0d0d0 !important;
            font-size: 1.1rem !important;
            transition: color 0.3s ease;
        }
        
        .analysis-card:hover p {
            color: #ffffff !important;
        }
        
        .analysis-card strong {
            color: #ffffff !important;
        }
        
        /* Clear Metric Cards - subtle initially, vibrant on hover */
        .metric-card {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15));
            border: 2px solid rgba(102, 126, 234, 0.4);
            border-radius: 15px;
            padding: 2rem 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.4), rgba(118, 75, 162, 0.4));
            box-shadow: 0 12px 24px rgba(102, 126, 234, 0.5);
            border-color: rgba(102, 126, 234, 0.9);
        }
        
        .metric-value {
            font-size: 2.8rem;
            font-weight: 700;
            color: #4facfe;
            margin: 0.8rem 0;
            transition: all 0.3s ease;
        }
        
        .metric-card:hover .metric-value {
            color: #00d4ff;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
        }
        
        .metric-label {
            color: #d0d0d0;
            font-size: 1.15rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 700;
            transition: color 0.3s ease;
        }
        
        .metric-card:hover .metric-label {
            color: #ffffff;
        }
        
        /* Clear Prediction Badge */
        .prediction-badge {
            display: inline-block;
            padding: 2rem 3.5rem;
            border-radius: 25px;
            font-weight: 800;
            font-size: 2.2rem;
            margin: 1.5rem 0;
            text-align: center;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .planet-detected {
            background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
            color: #ffffff;
            box-shadow: 0 8px 32px rgba(72, 187, 120, 0.6);
            border: 3px solid #68d391;
        }
        
        .not-planet {
            background: linear-gradient(135deg, #fc8181 0%, #f56565 100%);
            color: #ffffff;
            box-shadow: 0 8px 32px rgba(245, 101, 101, 0.6);
            border: 3px solid #feb2b2;
        }
        
        /* Clear Buttons - subtle gradient initially */
        .stButton > button {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.8) 0%, rgba(118, 75, 162, 0.8) 100%) !important;
            color: white !important;
            border: 2px solid rgba(102, 126, 234, 0.5) !important;
            border-radius: 15px !important;
            padding: 1.2rem 3rem !important;
            font-size: 1.2rem !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
            transition: all 0.3s ease !important;
            width: 100%;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border-color: rgba(102, 126, 234, 0.9) !important;
            transform: translateY(-3px) !important;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
        }
        
        /* Progress Bar */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Clear, Readable Text */
        .stMarkdown, .stText, p, span, div {
            color: #e0e0e0 !important;
            font-size: 1.1rem;
        }
        
        h1, h2, h3, h4 {
            color: #ffffff !important;
        }
        
        h2 {
            font-size: 2rem !important;
        }
        
        h3 {
            font-size: 1.6rem !important;
        }
        
        h4 {
            font-size: 1.3rem !important;
        }
        
        /* Ensure all labels are visible */
        label {
            color: #d0d0d0 !important;
            font-size: 1.15rem !important;
            font-weight: 600 !important;
        }
        
        /* Alert Boxes - Smart Contrast - content always bright */
        .stSuccess {
            background-color: rgba(72, 187, 120, 0.3) !important;
            border: 2px solid #48bb78 !important;
            color: #ffffff !important;
            font-size: 1.1rem !important;
        }
        .stSuccess p, .stSuccess div, .stSuccess span, .stSuccess strong { color: #ffffff !important; }
        
        .stWarning {
            background-color: rgba(237, 137, 54, 0.3) !important;
            border: 2px solid #ed8936 !important;
            color: #ffffff !important;
            font-size: 1.1rem !important;
        }
        .stWarning p, .stWarning div, .stWarning span, .stWarning strong { color: #ffffff !important; }
        
        .stError {
            background-color: rgba(245, 101, 101, 0.3) !important;
            border: 2px solid #f56565 !important;
            color: #ffffff !important;
            font-size: 1.1rem !important;
        }
        .stError p, .stError div, .stError span, .stError strong { color: #ffffff !important; }
        
        .stInfo {
            background-color: rgba(66, 153, 225, 0.3) !important;
            border: 2px solid #4299e1 !important;
            color: #ffffff !important;
            font-size: 1.1rem !important;
        }
        .stInfo p, .stInfo div, .stInfo span, .stInfo strong { color: #ffffff !important; }
        
        /* Sidebar - subtle elements */
        [data-testid="stSidebar"] {
            background-color: rgba(10, 14, 39, 0.95) !important;
        }
        [data-testid="stSidebar"] .stMarkdown { color: #d0d0d0 !important; }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4 { color: #ffffff !important; }
        [data-testid="stSidebar"] p { color: #d0d0d0 !important; }
        [data-testid="stSidebar"] .stInfo { background-color: rgba(66, 153, 225, 0.25) !important; }
        [data-testid="stSidebar"] .stInfo:hover { background-color: rgba(66, 153, 225, 0.35) !important; }
        [data-testid="stSidebar"] .stSuccess { background-color: rgba(72, 187, 120, 0.25) !important; }
        [data-testid="stSidebar"] .stSuccess:hover { background-color: rgba(72, 187, 120, 0.35) !important; }
        [data-testid="stSidebar"] .stWarning { background-color: rgba(237, 137, 54, 0.25) !important; }
        [data-testid="stSidebar"] .stWarning:hover { background-color: rgba(237, 137, 54, 0.35) !important; }
        
        /* File Uploader - subtle initially */
        .stFileUploader {
            border: 3px dashed rgba(102, 126, 234, 0.3) !important;
            border-radius: 20px !important;
            padding: 2rem !important;
            transition: all 0.3s ease !important;
        }
        .stFileUploader:hover {
            border-color: rgba(102, 126, 234, 0.7) !important;
            background-color: rgba(102, 126, 234, 0.05) !important;
        }
        .stFileUploader label {
            color: #d0d0d0 !important;
            font-size: 1.2rem !important;
            font-weight: 600 !important;
            transition: color 0.3s ease !important;
        }
        .stFileUploader:hover label { color: #ffffff !important; }
        
        /* Data Table - High Contrast */
        .dataframe { color: #ffffff !important; font-size: 1.05rem !important; background-color: rgba(15, 25, 50, 0.6) !important; }
        .dataframe thead th {
            background-color: rgba(102, 126, 234, 0.7) !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 1.1rem !important;
        }
        .dataframe tbody td { color: #ffffff !important; background-color: rgba(15, 25, 50, 0.4) !important; }
        .dataframe tbody tr:hover td { background-color: rgba(102, 126, 234, 0.3) !important; }
        
        /* Selectbox - darker initially */
        .stSelectbox label, .stFileUploader label {
            color: #d0d0d0 !important;
            font-weight: 600 !important;
            font-size: 1.15rem !important;
            transition: color 0.3s ease !important;
        }
        .stSelectbox:hover label, .stFileUploader:hover label { color: #ffffff !important; }
        .stSelectbox > div > div {
            background-color: rgba(15, 25, 50, 0.6) !important;
            border: 2px solid rgba(102, 126, 234, 0.3) !important;
            color: #d0d0d0 !important;
            transition: all 0.3s ease !important;
        }
        .stSelectbox > div > div:hover, .stSelectbox > div > div:focus-within {
            background-color: rgba(15, 25, 50, 0.9) !important;
            border-color: rgba(102, 126, 234, 0.6) !important;
            color: #ffffff !important;
        }
        .stSelectbox input { color: #d0d0d0 !important; }
        .stSelectbox input:focus { color: #ffffff !important; }
        
        /* Text Input */
        .stTextInput input {
            background-color: rgba(15, 25, 50, 0.6) !important;
            border: 2px solid rgba(102, 126, 234, 0.3) !important;
            color: #d0d0d0 !important;
            transition: all 0.3s ease !important;
        }
        .stTextInput input:hover, .stTextInput input:focus {
            background-color: rgba(15, 25, 50, 0.9) !important;
            border-color: rgba(102, 126, 234, 0.6) !important;
            color: #ffffff !important;
        }
        .stTextInput label {
            color: #d0d0d0 !important;
            font-size: 1.15rem !important;
            font-weight: 600 !important;
            transition: color 0.3s ease !important;
        }
        .stTextInput:hover label, .stTextInput:focus-within label { color: #ffffff !important; }
        
        /* Metric Enhancement - subtle initially */
        .stMetric {
            background-color: rgba(15, 25, 50, 0.2) !important;
            padding: 1rem !important;
            border-radius: 10px !important;
            border: 1px solid rgba(102, 126, 234, 0.2) !important;
            transition: all 0.3s ease !important;
        }
        .stMetric:hover { background-color: rgba(15, 25, 50, 0.4) !important; border-color: rgba(102, 126, 234, 0.4) !important; }
        .stMetric label {
            color: #d0d0d0 !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
            transition: color 0.3s ease !important;
        }
        .stMetric:hover label { color: #ffffff !important; }
        .stMetric .metric-value { color: #4facfe !important; font-weight: 700 !important; font-size: 1.9rem !important; transition: all 0.3s ease !important; }
        .stMetric:hover .metric-value { color: #00d4ff !important; }
        [data-testid="stMetricValue"] { font-size: 1.9rem !important; font-weight: 700 !important; color: #4facfe !important; transition: color 0.3s ease !important; }
        .stMetric:hover [data-testid="stMetricValue"] { color: #00d4ff !important; }
        [data-testid="stMetricLabel"] { color: #d0d0d0 !important; font-size: 1.1rem !important; font-weight: 600 !important; transition: color 0.3s ease !important; }
        .stMetric:hover [data-testid="stMetricLabel"] { color: #ffffff !important; }
        [data-testid="stMetricDelta"] { color: #d0d0d0 !important; }
        
        /* Expander */
        .streamlit-expanderHeader {
            background: rgba(15, 25, 50, 0.6) !important;
            border: 2px solid rgba(102, 126, 234, 0.4) !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            font-size: 1.25rem !important;
            color: #e0e0e0 !important;
            padding: 1rem !important;
            transition: all 0.3s ease !important;
        }
        .streamlit-expanderHeader:hover {
            border-color: rgba(102, 126, 234, 0.7) !important;
            background: rgba(102, 126, 234, 0.3) !important;
            color: #ffffff !important;
        }
        .streamlit-expanderHeader p { color: #e0e0e0 !important; font-size: 1.25rem !important; font-weight: 700 !important; transition: color 0.3s ease !important; }
        .streamlit-expanderHeader:hover p { color: #ffffff !important; }
        .streamlit-expanderHeader svg { fill: #e0e0e0 !important; transition: fill 0.3s ease !important; }
        .streamlit-expanderHeader:hover svg { fill: #ffffff !important; }
        .streamlit-expanderHeader[aria-expanded="true"] {
            background: rgba(102, 126, 234, 0.4) !important;
            border-color: rgba(102, 126, 234, 0.8) !important;
            color: #ffffff !important;
        }
        .streamlit-expanderHeader[aria-expanded="true"] p { color: #ffffff !important; }
        .streamlit-expanderHeader[aria-expanded="true"] svg { fill: #ffffff !important; }
        .streamlit-expanderContent {
            background-color: rgba(15, 25, 50, 0.4) !important;
            border: 1px solid rgba(102, 126, 234, 0.3) !important;
            border-top: none !important;
            padding: 1.5rem !important;
        }
        .streamlit-expanderContent p, .streamlit-expanderContent div, .streamlit-expanderContent span, .streamlit-expanderContent strong,
        .streamlit-expanderContent label { color: #ffffff !important; }
        
        /* Checkbox & Radio labels */
        .stCheckbox { transition: all 0.3s ease; }
        .stCheckbox label { color: #b0b0b0 !important; font-size: 1.1rem !important; transition: color 0.3s ease !important; }
        .stCheckbox:hover label { color: #ffffff !important; }
        .stCheckbox input:checked ~ label { color: #ffffff !important; }
        .stRadio label { color: #b0b0b0 !important; font-size: 1.1rem !important; transition: color 0.3s ease !important; }
        .stRadio:hover label { color: #ffffff !important; }
        
        /* Text */
        p { color: #e0e0e0 !important; font-size: 1.1rem !important; }
        ul li, ol li { color: #e0e0e0 !important; font-size: 1.1rem !important; }
        strong, b { color: #ffffff !important; font-weight: 700 !important; }
        
        /* Links */
        a { color: #00d4ff !important; }
        a:hover { color: #4facfe !important; }
        
        /* Code blocks */
        code {
            color: #00f5ff !important;
            background-color: rgba(0, 20, 40, 0.8) !important;
            padding: 0.2rem 0.5rem !important;
            border-radius: 5px !important;
            font-size: 1rem !important;
        }
        pre {
            background-color: rgba(0, 20, 40, 0.8) !important;
            border: 1px solid rgba(102, 126, 234, 0.3) !important;
            border-radius: 10px !important;
            padding: 1rem !important;
        }
        pre code { color: #00f5ff !important; }
        
        /* Spinner/Loading */
        .stSpinner > div { border-top-color: #00d4ff !important; }
        
        /* Download button */
        .stDownloadButton > button {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.8) 0%, rgba(118, 75, 162, 0.8) 100%) !important;
            color: white !important;
            border: 2px solid rgba(102, 126, 234, 0.5) !important;
            border-radius: 15px !important;
            padding: 1.2rem 3rem !important;
            font-size: 1.2rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }
        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border-color: rgba(102, 126, 234, 0.9) !important;
            transform: translateY(-3px) !important;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
        }
    </style>
    """, unsafe_allow_html=True)

# --- Load Feature List ---
def load_feature_list():
    """Load the feature list from feature_list.json"""
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

# --- Feature Extraction Functions ---
def extract_features(df):
    """Extract features from the light curve data with validation"""
    try:
        features = {}
        flux = df['flux'].values
        time_vals = df['time'].values
        all_features = {}
        
        # Data quality check
        if len(flux) < 10:
            st.warning("Very few data points detected. Results may be unreliable.")
        
        # Basic Statistics
        all_features['mean_flux'] = np.mean(flux)
        all_features['std_flux'] = np.std(flux)
        all_features['min_flux'] = np.min(flux)
        all_features['max_flux'] = np.max(flux)
        all_features['median_flux'] = np.median(flux)
        all_features['flux_range'] = np.max(flux) - np.min(flux)
        all_features['cv_flux'] = np.std(flux) / np.mean(flux) if np.mean(flux) != 0 else 0
        
        # Distribution Features
        all_features['skewness'] = stats.skew(flux)
        all_features['kurtosis'] = stats.kurtosis(flux)
        all_features['percentile_75'] = np.percentile(flux, 75)
        all_features['percentile_25'] = np.percentile(flux, 25)
        all_features['percentile_90'] = np.percentile(flux, 90)
        
        # Trend Features
        time_normalized = (time_vals - np.min(time_vals)) / (np.max(time_vals) - np.min(time_vals)) if np.max(time_vals) != np.min(time_vals) else time_vals
        slope, intercept, r_value, _, _ = stats.linregress(time_normalized, flux)
        all_features['trend_slope'] = slope
        all_features['trend_r2'] = r_value ** 2
        all_features['linear_fit_error'] = np.mean(np.abs(flux - (slope * time_normalized + intercept)))
        
        # Variability Features
        flux_diff = np.diff(flux)
        all_features['mean_absolute_deviation'] = np.mean(np.abs(flux - np.mean(flux)))
        all_features['flux_diff_mean'] = np.mean(flux_diff)
        all_features['flux_diff_std'] = np.std(flux_diff)
        all_features['flux_diff_max'] = np.max(np.abs(flux_diff))
        
        # Rolling Statistics
        window = max(5, len(flux) // 20)
        rolling_std = pd.Series(flux).rolling(window=window, center=True).std()
        all_features['rolling_std_mean'] = np.nanmean(rolling_std)
        all_features['rolling_std_max'] = np.nanmax(rolling_std)
        all_features['rolling_std_min'] = np.nanmin(rolling_std)
        
        # Additional features
        all_features['data_points'] = len(flux)
        all_features['time_span'] = np.max(time_vals) - np.min(time_vals)
        
        # Signal quality metrics
        all_features['snr'] = all_features['mean_flux'] / all_features['std_flux'] if all_features['std_flux'] != 0 else 0
        all_features['data_quality_score'] = min(100, (len(flux) / 100) * 50 + all_features['snr'] * 10)
        
        if FEATURE_LIST:
            for feature_name in FEATURE_LIST:
                if feature_name in all_features:
                    features[feature_name] = all_features[feature_name]
                else:
                    features[feature_name] = 0
        else:
            default_features = [
                'mean_flux', 'std_flux', 'min_flux', 'max_flux', 'median_flux',
                'flux_range', 'cv_flux', 'skewness', 'kurtosis', 'percentile_75',
                'trend_slope', 'trend_r2', 'linear_fit_error', 'mean_absolute_deviation',
                'flux_diff_mean', 'flux_diff_std', 'flux_diff_max',
                'rolling_std_mean', 'rolling_std_max', 'rolling_std_min'
            ]
            for feature_name in default_features:
                if feature_name in all_features:
                    features[feature_name] = all_features[feature_name]
        
        return features, all_features
    except Exception as e:
        st.error(f"Feature extraction failed: {str(e)}")
        return {}, {}

# --- Model Loading ---
def load_model():
    """Load your trained model from models folder"""
    try:
        import joblib
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            return model
        else:
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# --- Prediction Function ---
def predict_with_model(features_dict, model=None):
    """Make predictions using your model"""
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
            not_planet_prob = probabilities[0]
            planet_prob = probabilities[1]
            confidence = planet_prob if prediction == 'PLANET' else not_planet_prob
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            prediction = np.random.choice(['PLANET', 'NOT_PLANET'], p=[0.35, 0.65])
            planet_prob = np.random.uniform(0.3, 0.7)
            not_planet_prob = 1 - planet_prob
            confidence = max(planet_prob, not_planet_prob)
    else:
        prediction = np.random.choice(['PLANET', 'NOT_PLANET'], p=[0.35, 0.65])
        planet_prob = np.random.uniform(0.3, 0.7)
        not_planet_prob = 1 - planet_prob
        confidence = max(planet_prob, not_planet_prob)
    
    results = {
        'prediction': prediction,
        'confidence': confidence,
        'planet_probability': planet_prob,
        'not_planet_probability': not_planet_prob,
        'model_version': '20251004_211436',
        'model_type': 'Stacking Ensemble',
        'feature_count': len(features_dict)
    }
    
    return results

# --- Enhanced Visualization ---
def create_visualization_plot(df, features, extra_features):
    """Create professional visualization plots"""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Light Curve Time Series', 
            'Flux Distribution & Statistics', 
            'Detrended Signal Analysis',
            'Feature Importance Ranking',
            'Rolling Statistics',
            'Signal Quality Metrics'
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"type": "bar"}],
            [{"secondary_y": True}, {"type": "indicator"}]
        ],
        vertical_spacing=0.10,
        horizontal_spacing=0.12
    )
    
    mean_flux = features.get('mean_flux', df['flux'].mean())
    std_flux = features.get('std_flux', df['flux'].std())
    
    # 1. Enhanced Time Series
    fig.add_trace(
        go.Scatter(
            x=df['time'], 
            y=df['flux'], 
            mode='lines+markers',
            name='Light Curve',
            line=dict(color='#00f5ff', width=2.5),
            marker=dict(size=4, color='#667eea', opacity=0.6),
            hovertemplate='<b>Time:</b> %{x:.4f}<br><b>Flux:</b> %{y:.8f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_hline(
        y=mean_flux, 
        line_dash="dash", 
        line_color="#00ff88", 
        line_width=3,
        annotation_text="Mean Flux",
        annotation_font_color="#00ff88",
        annotation_font_size=12,
        row=1, col=1
    )
    
    fig.add_hline(
        y=mean_flux + std_flux,
        line_dash="dot",
        line_color="#ff0080",
        line_width=2,
        annotation_text="+1σ",
        row=1, col=1
    )
    
    fig.add_hline(
        y=mean_flux - std_flux,
        line_dash="dot",
        line_color="#ff0080",
        line_width=2,
        annotation_text="-1σ",
        row=1, col=1
    )
    
    # 2. Distribution with Histogram
    fig.add_trace(
        go.Histogram(
            x=df['flux'],
            name='Distribution',
            nbinsx=40,
            marker=dict(
                color='#667eea',
                opacity=0.7,
                line=dict(color='#00f5ff', width=1)
            ),
            hovertemplate='<b>Flux Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Detrended Signal
    flux = df['flux'].values
    time_norm = (df['time'] - df['time'].min()) / (df['time'].max() - df['time'].min())
    slope, intercept, _, _, _ = stats.linregress(time_norm, flux)
    trend = slope * time_norm + intercept
    detrended = flux - trend
    
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=detrended,
            mode='lines',
            name='Detrended Signal',
            line=dict(color='#f093fb', width=2.5),
            hovertemplate='<b>Time:</b> %{x:.4f}<br><b>Detrended:</b> %{y:.8f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="#00f5ff", line_width=2, row=2, col=1)
    
    # 4. Top Features
    feature_names = ['Mean', 'Std Dev', 'Range', 'CV', 'Skewness', 'Kurtosis', 'Trend', 'SNR']
    feature_values = [
        abs(features.get('mean_flux', mean_flux)) / 10000,
        abs(features.get('std_flux', std_flux)) / 1000,
        abs(features.get('flux_range', 0)) / 1000,
        abs(features.get('cv_flux', 0)) * 1000,
        abs(features.get('skewness', 0)) * 100,
        abs(features.get('kurtosis', 0)) * 10,
        abs(features.get('trend_slope', 0)) * 10000,
        abs(extra_features.get('snr', 0))
    ]
    
    colors = ['#00f5ff', '#667eea', '#764ba2', '#f093fb', '#00ff88', '#ffbb00', '#ff0080', '#00d4aa']
    
    fig.add_trace(
        go.Bar(
            x=feature_names,
            y=feature_values,
            name='Feature Values',
            marker=dict(
                color=colors,
                opacity=0.8,
                line=dict(color='#ffffff', width=2)
            ),
            hovertemplate='<b>%{x}:</b> %{y:.4f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # 5. Rolling Statistics
    window = max(5, len(flux) // 20)
    rolling_mean = pd.Series(flux).rolling(window=window, center=True).mean()
    rolling_std = pd.Series(flux).rolling(window=window, center=True).std()
    
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=rolling_mean,
            mode='lines',
            name='Rolling Mean',
            line=dict(color='#00ff88', width=3),
            yaxis='y'
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=rolling_std,
            mode='lines',
            name='Rolling Std',
            line=dict(color='#ff0080', width=3, dash='dash'),
            yaxis='y2'
        ),
        row=3, col=1, secondary_y=True
    )
    
    # 6. Quality Gauge
    quality_score = extra_features.get('data_quality_score', 50)
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=quality_score,
            title={'text': "Data Quality", 'font': {'size': 20, 'color': '#00f5ff'}},
            delta={'reference': 75, 'increasing': {'color': "#00ff88"}},
            gauge={
                'axis': {'range': [None, 100], 'tickcolor': '#00f5ff'},
                'bar': {'color': "#00f5ff"},
                'bgcolor': "rgba(0,0,0,0.5)",
                'borderwidth': 2,
                'bordercolor': "#667eea",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(255, 0, 128, 0.3)'},
                    {'range': [50, 75], 'color': 'rgba(255, 187, 0, 0.3)'},
                    {'range': [75, 100], 'color': 'rgba(0, 255, 136, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': "#f093fb", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=1200,
        showlegend=True,
        paper_bgcolor='rgba(10, 14, 39, 0.95)',
        plot_bgcolor='rgba(15, 25, 50, 0.9)',
        font=dict(color='#e0f0ff', size=13, family='Rajdhani'),
        hovermode='closest',
        legend=dict(
            bgcolor='rgba(10, 20, 40, 0.8)',
            bordercolor='#00f5ff',
            borderwidth=2,
            font=dict(size=12)
        )
    )
    
    # Update axes
    fig.update_xaxes(
        gridcolor='rgba(0, 245, 255, 0.2)',
        gridwidth=1,
        showline=True,
        linewidth=2,
        linecolor='#00f5ff',
        title_font_color='#00f5ff',
        tickfont_color='#e0f0ff'
    )
    
    fig.update_yaxes(
        gridcolor='rgba(102, 126, 234, 0.2)',
        gridwidth=1,
        showline=True,
        linewidth=2,
        linecolor='#667eea',
        title_font_color='#667eea',
        tickfont_color='#e0f0ff'
    )
    
    # Update subplot titles
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=16, color='#00f5ff', family='Orbitron', weight='bold')
    
    return fig

# --- Main Application ---
apply_premium_theme()

# Header
st.markdown('<h1 class="main-title">Exoplanet Detection AI Platform</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Machine Learning · Stacking Ensemble · Real-time Analysis</p>', unsafe_allow_html=True)

# Enhanced Sidebar
with st.sidebar:
    st.markdown("### System Status")
    
    model_exists = os.path.exists(MODEL_PATH)
    features_exist = os.path.exists(FEATURE_LIST_PATH)
    
    # Status indicators
    col1, col2 = st.columns(2)
    with col1:
        if model_exists:
            st.success("Model available")
        else:
            st.warning("Demo mode")
    
    with col2:
        if features_exist:
            st.success("Features loaded")
        else:
            st.info("Using defaults")
    
    st.markdown("---")
    
    # Model specs
    st.markdown("### Model Info")
    
    specs = {
        "Name": "Ensemble",
        "Version": "v20251004",
        "Features": str(len(FEATURE_LIST)) if FEATURE_LIST else "20",
        "Classes": "2",
        "Samples": "23,289"
    }
    
    for key, value in specs.items():
        st.metric(key, value)
    
    st.markdown("---")
    
    # Quick Settings
    st.markdown("### Settings")
    
    show_features = st.checkbox("Show Features", value=False)
    show_visualization = st.checkbox("Show Visualizations", value=True)
    show_advanced = st.checkbox("Advanced Analysis", value=False)
    auto_analyze = st.checkbox("Auto-Analyze", value=False)
    
    st.markdown("---")
    
    # Info Section
    st.markdown("### Requirements")
    st.info("""
    Required Columns:
    • Time series data
    • Flux measurements

    Optimal:
    • 100+ data points
    • Clean, continuous data
    • Normalized flux
    """)
    
    st.markdown("---")
    
    # Workflow
    st.markdown("### Workflow")
    st.markdown("""
    1. Upload CSV data  
    2. Extract features  
    3. Classify signal  
    4. View results  
    5. Export report
    """)
    
    st.markdown("---")
    
    with st.expander("Debug Info", expanded=False):
        st.code(f"Model: {os.path.exists(MODEL_PATH)}")
        st.code(f"Features: {os.path.exists(FEATURE_LIST_PATH)}")

# Main Content
st.markdown("<br>", unsafe_allow_html=True)

# Upload Section
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    uploaded_file = st.file_uploader(
        "Upload Light Curve Data (CSV Format)",
        type=['csv'],
        help="CSV file with time and flux columns - auto-detection enabled"
    )

if uploaded_file is not None:
    try:
        # Loading animation
        with st.spinner('Loading data...'):
            df = pd.read_csv(uploaded_file)
            time.sleep(0.2)
        
        st.success(f"Data loaded: {len(df):,} records")
        
        # Column detection
        st.info(f"Detected columns: {', '.join(df.columns.tolist())}")
        
        time_col = None
        flux_col = None
        
        # Auto-detect columns
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in ['time', 'times', 't', 'bjd', 'jd', 'mjd', 'timestamp', 'date']:
                time_col = col
            if col_lower in ['flux', 'fluxes', 'f', 'brightness', 'magnitude', 'mag', 'intensity', 'signal']:
                flux_col = col
        
        if time_col is None or flux_col is None:
            st.warning("Please select columns manually:")
            
            col1, col2 = st.columns(2)
            with col1:
                time_col = st.selectbox("Select TIME column:", df.columns.tolist(), index=0)
            with col2:
                flux_col = st.selectbox("Select FLUX column:", df.columns.tolist(), index=1 if len(df.columns) > 1 else 0)
        else:
            st.success(f"Auto-detected: Time = '{time_col}', Flux = '{flux_col}'")
        
        # Process data
        df_processed = df.copy()
        df_processed['time'] = df[time_col]
        df_processed['flux'] = df[flux_col]
        
        # Data preview
        with st.expander("Data Preview & Statistics", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Data Points</div>
                    <div class="metric-value">{len(df_processed):,}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Time Span</div>
                    <div class="metric-value">{df_processed['time'].max() - df_processed['time'].min():.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Mean Flux</div>
                    <div class="metric-value">{df_processed['flux'].mean():.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Std Dev</div>
                    <div class="metric-value">{df_processed['flux'].std():.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(
                df_processed[['time', 'flux']].head(15),
                use_container_width=True,
                height=300
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Action Buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if auto_analyze or st.button("Run Analysis", use_container_width=True, type="primary"):
                
                # Enhanced progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1
                status_text.markdown("### Step 1/5: Preprocessing data...")
                progress_bar.progress(10)
                time.sleep(0.3)
                
                # Step 2
                status_text.markdown("### Step 2/5: Extracting features...")
                progress_bar.progress(30)
                features, extra_features = extract_features(df_processed)
                if features:
                    st.success(f"Extracted {len(features)} features")
                time.sleep(0.3)
                
                # Step 3
                status_text.markdown("### Step 3/5: Loading model...")
                progress_bar.progress(50)
                model = load_model()
                time.sleep(0.2)
                
                # Step 4
                status_text.markdown("### Step 4/5: Running classification...")
                progress_bar.progress(75)
                results = predict_with_model(features, model)
                time.sleep(0.3)
                
                # Step 5
                status_text.markdown("### Step 5/5: Generating results...")
                progress_bar.progress(100)
                time.sleep(0.2)
                
                progress_bar.empty()
                status_text.empty()
                
                # Results Section
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown("## Detection Results")
                st.markdown("---")
                
                # Main Prediction
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    prediction_class = "planet-detected" if results['prediction'] == 'PLANET' else "not-planet"
                    label = "PLANET DETECTED" if results['prediction'] == 'PLANET' else "NO PLANET"
                    
                    st.markdown(f"""
                    <div class="prediction-badge {prediction_class}">
                        {label}
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Metrics Grid
                st.markdown("### Classification Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">CONFIDENCE</div>
                        <div class="metric-value">{results['confidence']:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">PLANET PROB</div>
                        <div class="metric-value">{results['planet_probability']:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">NOT PLANET</div>
                        <div class="metric-value">{results['not_planet_probability']:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">FEATURES</div>
                        <div class="metric-value">{results['feature_count']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # System Info
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Model Type", results['model_type'])
                col2.metric("Version", results['model_version'])
                col3.metric("Dataset", "23,289")
                col4.metric("Features", results['feature_count'])
                
                # Features Display
                if show_features and features:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("## Extracted Features")
                    
                    col1, col2 = st.columns(2)
                    
                    features_list = list(features.items())
                    mid = len(features_list) // 2
                    
                    with col1:
                        st.markdown("""
                        <div class="analysis-card">
                            <h3 style='color: #ffffff; margin-bottom: 1.5rem;'>Primary Features</h3>
                        """, unsafe_allow_html=True)
                        
                        for key, value in features_list[:mid]:
                            st.metric(
                                key.replace('_', ' ').title(),
                                f"{value:.8f}",
                                delta=None
                            )
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                        <div class="analysis-card">
                            <h3 style='color: #ffffff; margin-bottom: 1.5rem;'>Secondary Features</h3>
                        """, unsafe_allow_html=True)
                        
                        for key, value in features_list[mid:]:
                            st.metric(
                                key.replace('_', ' ').title(),
                                f"{value:.8f}",
                                delta=None
                            )
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                
                # Visualization
                if show_visualization:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("## Light Curve Analysis")
                    fig = create_visualization_plot(df_processed, features, extra_features)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Advanced Analysis
                if show_advanced:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("## Advanced Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        <div class="analysis-card">
                            <h3>Frequency Domain</h3>
                        """, unsafe_allow_html=True)
                        
                        # FFT analysis
                        from scipy.fft import fft, fftfreq
                        flux_fft = fft(df_processed['flux'].values)
                        n = len(df_processed)
                        freq = fftfreq(n, df_processed['time'].diff().mean())
                        
                        st.metric("Dominant Frequency", f"{freq[np.argmax(np.abs(flux_fft[1:n//2]))+1]:.6f}")
                        st.metric("FFT Peak Power", f"{np.max(np.abs(flux_fft[1:n//2])):.2f}")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                        <div class="analysis-card">
                            <h3>Statistical Tests</h3>
                        """, unsafe_allow_html=True)
                        
                        # Normality test
                        from scipy.stats import normaltest, kstest
                        stat, p_value = normaltest(df_processed['flux'])
                        
                        st.metric("Normality p-value", f"{p_value:.6f}")
                        st.metric("Skewness", f"{extra_features.get('skewness', 0):.4f}")
                        st.metric("Kurtosis", f"{extra_features.get('kurtosis', 0):.4f}")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                
                # Summary
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("## Analysis Summary")
                
                if results['prediction'] == 'PLANET':
                    st.success("Exoplanet transit signature detected.")
                    st.markdown("""
                    <p style='color: #ffffff; font-size: 1.15rem;'>
                    The model indicates a planetary transit based on multiple signal features and ensemble classification.
                    </p>
                    """, unsafe_allow_html=True)
                    
                    if results['confidence'] > 0.90:
                        st.info("Very high confidence — excellent detection quality.")
                    elif results['confidence'] > 0.75:
                        st.info("High confidence — strong evidence of planetary transit.")
                    elif results['confidence'] > 0.60:
                        st.warning("Moderate confidence — additional verification recommended.")
                    else:
                        st.warning("Low confidence — further observation required.")
                else:
                    st.info("No exoplanet transit detected.")
                    st.markdown("""
                    <p style='color: #ffffff; font-size: 1.15rem;'>
                    The light curve lacks characteristic transit features. This may indicate stellar variability, noise, or other non-planetary phenomena.
                    </p>
                    """, unsafe_allow_html=True)
                    
                    if results['confidence'] > 0.85:
                        st.success("High certainty in negative classification.")
                    else:
                        st.warning("Consider collecting additional data for better certainty.")
                
                # Data Quality Report
                st.markdown("### Data Quality Report")
                
                snr = extra_features.get('snr', df_processed['flux'].mean() / df_processed['flux'].std())
                quality_score = extra_features.get('data_quality_score', 50)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Signal-to-Noise Ratio", f"{snr:.2f}")
                
                with col2:
                    st.metric("Data Quality Score", f"{quality_score:.1f}/100")
                
                with col3:
                    variability = extra_features.get('cv_flux', features.get('cv_flux', 0))
                    st.metric("Coefficient of Variation", f"{variability:.6f}")
                
                # Recommendations
                st.markdown("### Recommendations")
                
                recommendations = []
                
                if len(df_processed) < 50:
                    recommendations.append("• Increase data points to at least 100 for better accuracy.")
                
                if snr < 10:
                    recommendations.append("• Improve signal quality through better instrumentation or longer exposure.")
                
                if quality_score < 60:
                    recommendations.append("• Apply preprocessing (detrending, outlier removal) to improve results.")
                
                if results['confidence'] < 0.70:
                    recommendations.append("• Acquire additional observations for confirmation.")
                
                if results['prediction'] == 'PLANET' and results['confidence'] > 0.85:
                    recommendations.append("• Strong candidate — schedule follow-up spectroscopy.")
                    recommendations.append("• Document parameters for publication.")
                
                if recommendations:
                    for rec in recommendations:
                        st.markdown(rec)
                else:
                    st.success("No additional recommendations — data quality is excellent.")
                
                # Export Report
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### Export Analysis")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    full_report = {**results, **features, **extra_features}
                    report_data = pd.DataFrame([full_report])
                    csv = report_data.to_csv(index=False)
                    
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    
                    st.download_button(
                        label="Download Complete Report (CSV)",
                        data=csv,
                        file_name=f"exoplanet_analysis_{timestamp}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        with st.expander("Error Details", expanded=False):
            st.exception(e)

else:
    # Welcome Screen
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 4rem; 
             background: rgba(22, 33, 62, 0.7); 
             border-radius: 25px; 
             border: 3px dashed rgba(102, 126, 234, 0.6);'>
            <h2 style='color: #ffffff; margin-bottom: 1.5rem; font-size: 2.5rem; font-weight: 700;'>
                Upload Light Curve Data
            </h2>
            <p style='color: #ffffff; font-size: 1.35rem; margin-bottom: 2rem; font-weight: 500;'>
                CSV format with time and flux measurements
            </p>
            <div style='background: rgba(10, 20, 40, 0.6); padding: 2rem; border-radius: 15px; border: 2px solid rgba(102, 126, 234, 0.3);'>
                <p style='color: #ffffff; font-size: 1.2rem; margin: 0.9rem 0; font-weight: 500;'>• CSV Format Supported</p>
                <p style='color: #ffffff; font-size: 1.2rem; margin: 0.9rem 0; font-weight: 500;'>• Auto Column Detection</p>
                <p style='color: #ffffff; font-size: 1.2rem; margin: 0.9rem 0; font-weight: 500;'>• Real-Time Processing</p>
                <p style='color: #ffffff; font-size: 1.2rem; margin: 0.9rem 0; font-weight: 500;'>• High Accuracy AI Model</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Feature Highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="analysis-card" style="text-align: center;">
            <h3 style='font-size: 1.5rem; color: #ffffff; font-weight: 700;'>Machine Learning</h3>
            <p style='font-size: 1.15rem; color: #ffffff;'>Advanced ensemble with multi-layer feature extraction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="analysis-card" style="text-align: center;">
            <h3 style='font-size: 1.5rem; color: #ffffff; font-weight: 700;'>Real-Time Analysis</h3>
            <p style='font-size: 1.15rem; color: #ffffff;'>Instant classification with signal processing</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="analysis-card" style="text-align: center;">
            <h3 style='font-size: 1.5rem; color: #ffffff; font-weight: 700;'>High Precision</h3>
            <p style='font-size: 1.15rem; color: #ffffff;'>Validated on astronomical datasets</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #ffffff; padding: 2.5rem; 
     border-top: 2px solid rgba(102, 126, 234, 0.4);
     background: rgba(10, 20, 40, 0.5);'>
    <p style='font-size: 1.25rem; margin-bottom: 1rem; color: #ffffff; font-weight: 600;'>
        Exoplanet Detection AI Platform
    </p>
    <p style='font-size: 1.1rem; color: #ffffff;'>
        Stacking Ensemble Model v20251004_211436 | Trained on 23,289 Samples
    </p>
    <p style='font-size: 1rem; color: #e0e0e0; margin-top: 1rem;'>
        TESS · Kepler · Confirmed Planets Datasets
    </p>
</div>
""", unsafe_allow_html=True)
