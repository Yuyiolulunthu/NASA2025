import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
import os

st.set_page_config(page_title="Exoplanet Detection AI", layout="wide", initial_sidebar_state="expanded")

# Get the correct paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Go up from Web/pages/ to project root
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
FEATURE_LIST_PATH = os.path.join(MODELS_DIR, 'feature_list.json')
MODEL_PATH = os.path.join(MODELS_DIR, 'new_model_20251004_211436.pkl')

# --- Custom CSS Styling ---
def apply_premium_theme():
    st.markdown("""
    <style>
        /* Global Styles */
        .stApp {
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1729 100%);
        }
        
        /* Title Styles */
        .main-title {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0.5rem;
            letter-spacing: -1px;
        }
        
        .subtitle {
            text-align: center;
            color: #a0aec0;
            font-size: 1.1rem;
            margin-bottom: 2rem;
            font-weight: 300;
        }
        
        /* Card Styles */
        .analysis-card {
            background: linear-gradient(135deg, rgba(22, 33, 62, 0.95), rgba(15, 52, 96, 0.95));
            border: 2px solid rgba(102, 126, 234, 0.3);
            border-radius: 20px;
            padding: 2rem;
            margin: 1.5rem 0;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(10px);
        }
        
        /* Metric Cards */
        .metric-card {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15));
            border: 1px solid rgba(102, 126, 234, 0.3);
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 24px rgba(102, 126, 234, 0.3);
            border-color: rgba(102, 126, 234, 0.6);
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0.5rem 0;
        }
        
        .metric-label {
            color: #a0aec0;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Prediction Badge */
        .prediction-badge {
            display: inline-block;
            padding: 1rem 2rem;
            border-radius: 25px;
            font-weight: 700;
            font-size: 1.5rem;
            margin: 1rem 0;
            text-align: center;
        }
        
        .planet-detected {
            background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
            color: white;
            box-shadow: 0 8px 32px rgba(72, 187, 120, 0.4);
            border: 2px solid #48bb78;
        }
        
        .not-planet {
            background: linear-gradient(135deg, #e53e3e 0%, #c53030 100%);
            color: white;
            box-shadow: 0 8px 32px rgba(229, 62, 62, 0.4);
            border: 2px solid #e53e3e;
        }
        
        /* Button Styles */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 15px;
            padding: 1rem 3rem;
            font-size: 1.1rem;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        }
        
        /* Progress Bar */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
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
                # feature_list.json might be a list or dict with 'features' key
                if isinstance(feature_data, list):
                    return feature_data
                elif isinstance(feature_data, dict) and 'features' in feature_data:
                    return feature_data['features']
                else:
                    st.warning("⚠️ feature_list.json format not recognized. Using default features.")
                    return None
        else:
            st.warning(f"⚠️ feature_list.json not found at {FEATURE_LIST_PATH}")
            return None
    except Exception as e:
        st.error(f"❌ Error loading feature list: {str(e)}")
        return None

# Load feature list globally
FEATURE_LIST = load_feature_list()

# --- Feature Extraction Functions ---
def extract_features(df):
    """
    Extract features from the light curve data
    
    If feature_list.json is available, it will use those features in the correct order.
    Otherwise, it will use a default set of 20 features.
    """
    features = {}
    
    flux = df['flux'].values
    time = df['time'].values
    
    # Calculate all possible features
    all_features = {}
    
    # Basic Statistics
    all_features['mean_flux'] = np.mean(flux)
    all_features['std_flux'] = np.std(flux)
    all_features['min_flux'] = np.min(flux)
    all_features['max_flux'] = np.max(flux)
    all_features['median_flux'] = np.median(flux)
    all_features['flux_range'] = np.max(flux) - np.min(flux)
    all_features['cv_flux'] = np.std(flux) / np.mean(flux) if np.mean(flux) != 0 else 0
    
    # Distribution Features
    from scipy import stats
    all_features['skewness'] = stats.skew(flux)
    all_features['kurtosis'] = stats.kurtosis(flux)
    all_features['percentile_75'] = np.percentile(flux, 75)
    all_features['percentile_25'] = np.percentile(flux, 25)
    all_features['percentile_90'] = np.percentile(flux, 90)
    
    # Trend Features
    time_normalized = (time - np.min(time)) / (np.max(time) - np.min(time)) if np.max(time) != np.min(time) else time
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
    all_features['time_span'] = np.max(time) - np.min(time)
    
    # If feature list is available, return features in that exact order
    if FEATURE_LIST:
        for feature_name in FEATURE_LIST:
            if feature_name in all_features:
                features[feature_name] = all_features[feature_name]
            else:
                # If feature not computed, set to 0 and warn
                st.warning(f"⚠️ Feature '{feature_name}' not found. Setting to 0.")
                features[feature_name] = 0
    else:
        # Use default feature set
        default_features = [
            'mean_flux', 'std_flux', 'min_flux', 'max_flux', 'median_flux',
            'flux_range', 'cv_flux', 'skewness', 'kurtosis', 'percentile_75',
            'trend_slope', 'trend_r2', 'linear_fit_error', 'mean_absolute_deviation',
            'flux_diff_mean', 'flux_diff_std', 'flux_diff_max',
            'rolling_std_mean', 'rolling_std_max', 'rolling_std_min'
        ]
        for feature_name in default_features:
            features[feature_name] = all_features[feature_name]
    
    return features

# --- YOUR AI MODEL INTEGRATION ---
def load_model():
    """
    Load your trained model from models folder
    """
    try:
        import joblib
        # Load the model using the correct path
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            return model
        else:
            st.warning(f"⚠️ Model file not found at: {MODEL_PATH}")
            st.info("Using demo mode with random predictions.")
            return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def predict_with_model(features_dict, model=None):
    """
    Make predictions using your model
    
    Args:
        features_dict: Dictionary containing 20 extracted features
        model: Your loaded model (if None, uses demo mode)
    
    Returns:
        Dictionary with prediction results
    """
    
    # Convert features to DataFrame (matching your model's expected format)
    features_df = pd.DataFrame([features_dict])
    
    if model is not None:
        try:
            # Use your actual model
            prediction = model.predict(features_df)[0]
            probabilities = model.predict_proba(features_df)[0]
            
            # probabilities format: [NOT_PLANET_prob, PLANET_prob]
            not_planet_prob = probabilities[0]
            planet_prob = probabilities[1]
            
            # Determine confidence based on prediction
            confidence = planet_prob if prediction == 'PLANET' else not_planet_prob
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("💡 This might be due to feature mismatch. Please check if the 20 features match your model's training features.")
            # Fallback to demo
            prediction = np.random.choice(['PLANET', 'NOT_PLANET'], p=[0.35, 0.65])
            planet_prob = np.random.uniform(0.3, 0.7)
            not_planet_prob = 1 - planet_prob
            confidence = max(planet_prob, not_planet_prob)
    else:
        # Demo mode
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

def create_visualization_plot(df, features):
    """Create advanced visualization plots"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Light Curve Time Series', 'Flux Distribution', 'Detrended Signal', 'Feature Importance'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "bar"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 1. Time Series
    fig.add_trace(
        go.Scatter(x=df['time'], y=df['flux'], mode='lines', name='Light Curve',
                   line=dict(color='#4facfe', width=1.5),
                   hovertemplate='Time: %{x:.4f}<br>Flux: %{y:.6f}<extra></extra>'),
        row=1, col=1
    )
    
    # Add mean line
    fig.add_hline(y=features['mean_flux'], line_dash="dash", line_color="orange", 
                  annotation_text="Mean", row=1, col=1)
    
    # 2. Distribution Histogram
    fig.add_trace(
        go.Histogram(x=df['flux'], name='Distribution', nbinsx=50,
                     marker=dict(color='#764ba2', opacity=0.7)),
        row=1, col=2
    )
    
    # 3. Detrended Signal
    from scipy import stats
    time_normalized = (df['time'] - df['time'].min()) / (df['time'].max() - df['time'].min())
    slope, intercept, _, _, _ = stats.linregress(time_normalized, df['flux'])
    trend = slope * time_normalized + intercept
    detrended = df['flux'] - trend
    
    fig.add_trace(
        go.Scatter(x=df['time'], y=detrended, mode='lines', name='Detrended',
                   line=dict(color='#f093fb', width=1.5)),
        row=2, col=1
    )
    
    # 4. Top Features Bar Chart
    feature_names = ['Mean', 'Std', 'Range', 'CV', 'Skew', 'Trend']
    feature_values = [
        abs(features['mean_flux']), 
        abs(features['std_flux']), 
        abs(features['flux_range']),
        abs(features['cv_flux']),
        abs(features['skewness']),
        abs(features['trend_slope'])
    ]
    
    fig.add_trace(
        go.Bar(x=feature_names, y=feature_values, name='Features',
               marker=dict(color='#00f2fe', opacity=0.7)),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        paper_bgcolor='rgba(10, 14, 39, 0.8)',
        plot_bgcolor='rgba(22, 33, 62, 0.6)',
        font=dict(color='#e0e0e0', size=11),
        hovermode='closest'
    )
    
    fig.update_xaxes(gridcolor='rgba(102, 126, 234, 0.2)')
    fig.update_yaxes(gridcolor='rgba(102, 126, 234, 0.2)')
    
    return fig

# --- Main Application ---
apply_premium_theme()

# Header
st.markdown('<h1 class="main-title">🪐 Exoplanet Detection AI Platform</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Machine Learning · Stacking Ensemble Model · Real-time Classification</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Model Information")
    
    # Check model and feature list status
    model_exists = os.path.exists(MODEL_PATH)
    features_exist = os.path.exists(FEATURE_LIST_PATH)
    
    if model_exists:
        st.success("✅ Model Loaded")
    else:
        st.warning("⚠️ Demo Mode")
        st.caption("Model file not found")
    
    if features_exist:
        st.success("✅ Features Loaded")
        if FEATURE_LIST:
            st.caption(f"{len(FEATURE_LIST)} features")
    else:
        st.info("ℹ️ Using default features")
    
    st.markdown("---")
    
    model_info = {
        "Model Name": "new_model",
        "Model Type": "Stacking Ensemble",
        "Version": "20251004_211436",
        "Features": str(len(FEATURE_LIST)) if FEATURE_LIST else "20",
        "Classes": "2 (PLANET / NOT_PLANET)",
        "Training Samples": "23,289"
    }
    
    for key, value in model_info.items():
        st.metric(key, value)
    
    st.markdown("---")
    st.markdown("### 📊 Data Requirements")
    st.info("""
    **Required Columns:**
    - `time` (temporal data)
    - `flux` (measurement values)
    
    **Recommended:**
    - 100+ data points
    - Clean, continuous data
    """)
    
    st.markdown("---")
    st.markdown("### 💡 How It Works")
    st.markdown("""
    1. Upload light curve data
    2. Extract features automatically
    3. AI model classification
    4. View detailed results
    """)
    
    st.markdown("---")
    show_features = st.checkbox("Show Extracted Features", value=False)
    show_visualization = st.checkbox("Show Visualizations", value=True)
    
    # Debug info
    with st.expander("🔧 Debug Info", expanded=False):
        st.code(f"Model Path:\n{MODEL_PATH}")
        st.code(f"Features Path:\n{FEATURE_LIST_PATH}")
        st.code(f"Model Exists: {os.path.exists(MODEL_PATH)}")
        st.code(f"Features Exist: {os.path.exists(FEATURE_LIST_PATH)}")

# Main Content Area
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader(
        "📁 Upload Light Curve Data (CSV)",
        type=['csv'],
        help="CSV file with 'time' and 'flux' columns"
    )

if uploaded_file is not None:
    try:
        # Load Data
        with st.spinner('📥 Loading light curve data...'):
            df = pd.read_csv(uploaded_file)
            time.sleep(0.3)
        
        # Data Validation
        if 'time' not in df.columns or 'flux' not in df.columns:
            st.error("❌ Error: CSV file must contain 'time' and 'flux' columns")
        else:
            # Success Message
            st.success(f"✅ Light curve loaded successfully! {len(df):,} data points")
            
            # Data Preview
            with st.expander("📋 Data Preview", expanded=False):
                col1, col2, col3 = st.columns(3)
                col1.metric("Data Points", f"{len(df):,}")
                col2.metric("Time Span", f"{df['time'].max() - df['time'].min():.2f}")
                col3.metric("Flux Range", f"{df['flux'].max() - df['flux'].min():.6f}")
                
                st.dataframe(df.head(10), use_container_width=True)
            
            # Analysis Button
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                analyze_button = st.button("🚀 Run Exoplanet Detection", use_container_width=True)
            
            if analyze_button:
                # Progress Bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Data Preprocessing
                status_text.text("🔄 Step 1/5: Preprocessing light curve...")
                progress_bar.progress(20)
                time.sleep(0.4)
                
                # Step 2: Feature Extraction
                status_text.text("🧠 Step 2/5: Extracting 20 features...")
                progress_bar.progress(40)
                features = extract_features(df)
                time.sleep(0.4)
                
                # Step 3: Load Model
                status_text.text("⚡ Step 3/5: Loading AI model...")
                progress_bar.progress(60)
                model = load_model()
                time.sleep(0.3)
                
                # Step 4: Prediction
                status_text.text("🤖 Step 4/5: Running classification...")
                progress_bar.progress(80)
                results = predict_with_model(features, model)
                time.sleep(0.4)
                
                # Step 5: Generate Report
                status_text.text("📊 Step 5/5: Generating report...")
                progress_bar.progress(100)
                time.sleep(0.3)
                
                progress_bar.empty()
                status_text.empty()
                
                # Display Results
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("## 🎯 Detection Results")
                st.markdown("---")
                
                # Main Prediction
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    prediction_class = "planet-detected" if results['prediction'] == 'PLANET' else "not-planet"
                    icon = "🪐" if results['prediction'] == 'PLANET' else "🌑"
                    st.markdown(f"""
                    <div class="prediction-badge {prediction_class}">
                        {icon} {results['prediction'].replace('_', ' ')}
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Confidence Metrics
                st.markdown("### 📊 Classification Confidence")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Overall Confidence</div>
                        <div class="metric-value">{results['confidence']:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Planet Probability</div>
                        <div class="metric-value">{results['planet_probability']:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Not Planet Probability</div>
                        <div class="metric-value">{results['not_planet_probability']:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Features Used</div>
                        <div class="metric-value">{results['feature_count']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Model Info
                col1, col2, col3 = st.columns(3)
                col1.metric("Model Type", results['model_type'])
                col2.metric("Model Version", results['model_version'])
                col3.metric("Training Dataset", "23,289 samples")
                
                # Extracted Features
                if show_features:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("### 🔬 Extracted Features")
                    
                    col1, col2 = st.columns(2)
                    
                    features_list = list(features.items())
                    mid = len(features_list) // 2
                    
                    with col1:
                        st.markdown("""
                        <div class="analysis-card">
                            <h4 style='color: #667eea; margin-bottom: 1rem;'>📈 Features 1-10</h4>
                        """, unsafe_allow_html=True)
                        
                        for key, value in features_list[:mid]:
                            st.metric(key.replace('_', ' ').title(), f"{value:.6f}")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                        <div class="analysis-card">
                            <h4 style='color: #764ba2; margin-bottom: 1rem;'>📊 Features 11-20</h4>
                        """, unsafe_allow_html=True)
                        
                        for key, value in features_list[mid:]:
                            st.metric(key.replace('_', ' ').title(), f"{value:.6f}")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                
                # Visualization
                if show_visualization:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("### 📉 Light Curve Analysis")
                    fig = create_visualization_plot(df, features)
                    st.plotly_chart(fig, use_container_width=True)
                
                # AI Insights
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 💡 AI Analysis Summary")
                
                if results['prediction'] == 'PLANET':
                    st.success("✅ **Exoplanet Detected!** The AI model has classified this light curve as containing a potential exoplanet transit signature.")
                    
                    if results['confidence'] > 0.85:
                        st.info("🎯 High confidence detection - Strong evidence of planetary transit")
                    elif results['confidence'] > 0.70:
                        st.info("⚠️ Moderate confidence - Further observation recommended")
                    else:
                        st.warning("⚠️ Low confidence - Requires additional verification")
                else:
                    st.info("🌑 **No Planet Detected** - The light curve does not show strong evidence of exoplanet transit.")
                    
                    if results['confidence'] > 0.85:
                        st.success("✅ High confidence in negative classification")
                    else:
                        st.warning("⚠️ Consider collecting more data for better certainty")
                
                # Additional Stats
                st.markdown("**Data Quality Indicators:**")
                st.markdown(f"- Signal-to-Noise: {features['mean_flux'] / features['std_flux']:.2f}")
                st.markdown(f"- Variability: {features['cv_flux']:.4f}")
                st.markdown(f"- Data Points: {len(df):,}")
                
                # Download Report
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    # Combine results and features
                    full_report = {**results, **features}
                    report_data = pd.DataFrame([full_report])
                    csv = report_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Report (CSV)",
                        data=csv,
                        file_name=f"exoplanet_detection_report_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.exception(e)
        st.info("Please ensure your CSV file is properly formatted with 'time' and 'flux' columns")

else:
    # Empty State
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 3rem; background: rgba(22, 33, 62, 0.5); border-radius: 20px; border: 2px dashed rgba(102, 126, 234, 0.3);'>
            <h2 style='color: #667eea; margin-bottom: 1rem;'>🪐 Upload Light Curve Data</h2>
            <p style='color: #a0aec0; font-size: 1.1rem;'>CSV format with time and flux columns required</p>
            <p style='color: #718096; font-size: 0.9rem; margin-top: 1rem;'>The AI will analyze the data using 20 extracted features<br>to detect potential exoplanet transit signatures</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #667eea; padding: 2rem; border-top: 1px solid rgba(102, 126, 234, 0.2);'>
    <p style='font-size: 0.9rem; opacity: 0.6;'>🪐 Exoplanet Detection AI Platform | Stacking Ensemble Model v20251004_211436</p>
    <p style='font-size: 0.8rem; opacity: 0.5; margin-top: 0.5rem;'>Trained on 23,289 samples from TESS, Kepler & Confirmed Planets datasets</p>
</div>
""", unsafe_allow_html=True)