# Web/pages/about.py
import streamlit as st
import os

# ---- Banner (global) ----
try:
    from components.banner import render_banner
except Exception:
    render_banner = None

# ---- Page config ----
st.set_page_config(
    page_title="About Our Models — ExoMatch",
    page_icon="Web/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

if render_banner:
    render_banner()

# Hide Streamlit default chrome (match other pages)
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---- Premium theme (copied & aligned with your other pages) ----
def apply_premium_theme():
    st.markdown("""
    <style>
        /* Global */
        .stApp {
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1729 100%);
            color: #e0e0e0;
        }
        /* Title */
        .main-title {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3rem; font-weight: 800; text-align: center;
            margin-bottom: .4rem; letter-spacing: -1px;
        }
        .subtitle {
            text-align: center; color: #d0d0d0; font-size: 1.05rem;
            margin-bottom: 1.3rem; font-weight: 400;
        }
        /* Cards (same vibe as analysis/vetting) */
        .analysis-card {
            background: linear-gradient(135deg, rgba(22,33,62,.95), rgba(15,52,96,.95));
            border: 2px solid rgba(102,126,234,.5);
            border-radius: 20px; padding: 1.35rem 1.6rem; margin: 1rem 0 1.2rem 0;
            box-shadow: 0 8px 32px rgba(31,38,135,.5); backdrop-filter: blur(10px);
        }
        .analysis-card h4, .analysis-card h5 { color: #ffffff !important; margin: 0 0 .35rem 0; }
        .muted { color: #cbd5e1; }
        .kpi {
            display: inline-block; padding: .35rem .55rem; border-radius: 10px;
            background: linear-gradient(135deg, rgba(102,126,234,.25), rgba(118,75,162,.25));
            border: 1px solid rgba(102,126,234,.4); margin: .25rem .25rem 0 0;
            font-weight: 700; color: #b9c3ff; font-size: .9rem;
        }
        /* Table */
        table { width: 100%; border-collapse: collapse; }
        thead th {
            background: rgba(102,126,234,.25);
            border-bottom: 1px solid rgba(102,126,234,.45);
            color: #fff; text-align: left;
        }
        td, th { padding: .6rem .55rem; border-bottom: 1px solid rgba(102,126,234,.18); }
        /* Lists */
        .tight li { margin: .18rem 0; }
        /* Section header */
        .section-title {
            font-weight: 800; letter-spacing: .2px; margin: .2rem 0 .6rem 0;
        }
        /* Subtle dividers */
        .divider { height: 1px; background: rgba(102,126,234,.35); margin: 1.1rem 0; }
        /* Layout container width */
        .mx { max-width: 1150px; margin: 0 auto; }
    </style>
    """, unsafe_allow_html=True)

apply_premium_theme()

# ---- Page header ----
st.markdown('<div class="mx">', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">About Our Models</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Lightweight, explainable machine learning for professional exoplanet vetting</p>', unsafe_allow_html=True)

# ---- Papers section ----
st.markdown("### Papers We Build Upon")
st.markdown("""
<div class="analysis-card">
  <h4>Exoplanet Detection Using Machine Learning</h4>
  <p class="muted">
    Classical transit searches (e.g., BLS) degrade under low S/N and heavy systematics.
    Deep learning can recover sensitivity but often at higher compute cost and lower interpretability.
    This line of work motivates a lightweight ML stack that maintains performance while improving
    efficiency and explainability.
  </p>
  <h5>Method</h5>
  <ul class="tight">
    <li><b>Pre-processing</b> — Clean light curves: remove systematics, impute gaps, de-trend.</li>
    <li><b>Feature extraction</b> — ~789 descriptors via <code>tsfresh</code> (variance, energy, peaks, etc.).</li>
    <li><b>Classifier</b> — Gradient-boosted trees (e.g., <b>LightGBM</b>).</li>
    <li><b>Data</b> — Train/validate on <b>Kepler</b>; generalize on <b>TESS</b>; stress-test with simulated K2.</li>
  </ul>
  <h5>Reported Results</h5>
  <div>
    <span class="kpi">Kepler AUC ≈ 0.948</span>
    <span class="kpi">Kepler Recall ≈ 96%</span>
    <span class="kpi">TESS Accuracy ≈ 98%</span>
    <span class="kpi">TESS Recall ≈ 82%</span>
    <span class="kpi">TESS Precision ≈ 63%</span>
  </div>
  <div class="divider"></div>
  <h5>Pros</h5>
  <ul class="tight">
    <li>CPU-friendly, minutes to retrain; interpretable feature importance; mission-agnostic.</li>
  </ul>
  <h5>Cons</h5>
  <ul class="tight">
    <li>Below best deep nets at extreme low S/N; lower precision on imbalanced TESS; trees not fully transparent.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="analysis-card">
  <h4>Assessment of Ensemble-Based ML Algorithms for Exoplanet Identification</h4>
  <p class="muted">
    Ensemble strategies (bagging, boosting, stacking) improve robustness in noisy regimes and
    reduce false positives compared with single learners or pure BLS-style pipelines.
  </p>
  <h5>Method</h5>
  <ul class="tight">
    <li><b>Data</b> — Kepler light curves, de-trended & normalized, with positive/negative labels.</li>
    <li><b>Features</b> — Time-series statistics capturing amplitude, period, depth, symmetry, energy.</li>
    <li><b>Models</b> — Single learners (RF/GBM/XGBoost/SVM/DT) vs. <b>ensembles</b> (Voting/Bagging/Boosting/<b>Stacking</b>).</li>
  </ul>
  <h5>Reported Results</h5>
  <div>
    <span class="kpi">Accuracy (single): 95–97%</span>
    <span class="kpi">Accuracy (stacking): ~98.5%</span>
    <span class="kpi">Recall (ensemble): &gt; 90%</span>
    <span class="kpi">Precision (ensemble): 85–88%</span>
    <span class="kpi">F1 (ensemble): ~0.91–0.93</span>
    <span class="kpi">ROC–AUC (ensemble): ~0.97–0.98</span>
  </div>
  <div class="divider"></div>
  <h5>Key Findings</h5>
  <ul class="tight">
    <li>Stacking consistently outperforms single models in low S/N scenarios; period/depth/symmetry dominate importance.</li>
  </ul>
  <h5>Limitations</h5>
  <ul class="tight">
    <li>Higher training cost than single learners; class imbalance still challenging in edge cases.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

# ---- Comparative table ----
st.markdown("### Comparative Summary")
st.markdown("""
<div class="analysis-card">
<table>
  <thead>
    <tr>
      <th>Aspect</th>
      <th>“Exoplanet Detection using ML”</th>
      <th>“Ensemble-Based ML for Identification”</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Data</td>
      <td>Kepler (train/val), TESS (generalization), simulated K2</td>
      <td>Kepler mission light curves</td>
    </tr>
    <tr>
      <td>Core Method</td>
      <td>~789 <code>tsfresh</code> features; classifier: <b>LightGBM</b></td>
      <td>Multiple single models vs. <b>ensembles</b> (Voting/Bagging/Boosting/<b>Stacking</b>)</td>
    </tr>
    <tr>
      <td>Compute</td>
      <td>Lightweight; CPU minutes; no GPU required</td>
      <td>Higher than single learners; still lighter than deep nets</td>
    </tr>
    <tr>
      <td>Kepler Results</td>
      <td>AUC ≈ 0.948; Recall ≈ 96%</td>
      <td>Accuracy ≈ 98.5%; Recall &gt; 90%; Precision ≈ 85–88%; F1 ≈ 0.91–0.93; ROC–AUC ≈ 0.97–0.98</td>
    </tr>
    <tr>
      <td>Strengths</td>
      <td>Fast training; interpretable; mission-agnostic</td>
      <td>Higher accuracy; better precision/recall tradeoff; robust generalization</td>
    </tr>
    <tr>
      <td>Limitations</td>
      <td>Lower precision on imbalanced TESS; residual opacity</td>
      <td>More training time; imbalance sensitivity remains</td>
    </tr>
  </tbody>
</table>
</div>
""", unsafe_allow_html=True)

# ---- Our approach ----
st.markdown("### Our Approach: Pragmatic, Explainable Ensemble Pipeline")
st.markdown("""
<div class="analysis-card">
  <ol class="tight">
    <li><b>Feature extraction</b> — Compute hundreds of descriptors via <code>tsfresh</code>/<code>sktime</code>; augment with BLS/TLS period, depth, duration.</li>
    <li><b>Base learners</b> — LightGBM/XGBoost/CatBoost + Random Forest/ExtraTrees + a linear baseline (LogReg or SVM).</li>
    <li><b>Ensembling</b> — Start with weighted voting; adopt <b>stacking</b> using Logistic Regression/LightGBM on out-of-fold probabilities.</li>
    <li><b>Evaluation</b> — K-fold CV with ROC–AUC, PR–AUC, F1, and <i>Recall @ 1% FPR</i> benchmarked vs. BLS/TLS.</li>
  </ol>
  <p class="muted" style="margin:.4rem 0 0 0;">This blend keeps training cycles CPU-friendly, exposes feature importance for QA,
  and gains robustness from multiple learners—ideal for production vetting with human-in-the-loop review.</p>
</div>
""", unsafe_allow_html=True)

# ---- Advantages & Caveats ----
c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Advantages")
    st.markdown("""
- **Fast iteration** — Minutes to retrain on CPUs; easy CI/CD.
- **Explainability** — Global/per-sample importances support scientific QA.
- **Generalization** — Works across missions with consistent pre-processing.
- **Operational fit** — Rank-ordered candidates and uncertainty flags integrate with expert vetting.
""")
with c2:
    st.markdown("#### Caveats")
    st.markdown("""
- **Extreme low S/N** — Deep nets may still edge out ensembles.
- **Class imbalance** — Use `class_weight=balanced` / `scale_pos_weight`, calibrated thresholds, and PR–AUC tracking.
- **Residual opacity** — Tree ensembles remain partly black-box despite feature importance.
""")

# ---- Roadmap ----
st.markdown("### Roadmap")
st.markdown("""
<div class="analysis-card">
  <ul class="tight">
    <li><b>Unified multi-mission model</b> with domain adaptation to reduce calibration drift across Kepler/K2/TESS.</li>
    <li><b>Imbalance remedies</b> — augmentation, semi-supervised PU learning, calibrated thresholds per target class.</li>
    <li><b>Unsupervised discovery</b> — autoencoders/clustering to surface novel transit-like anomalies.</li>
    <li><b>Application to unconfirmed candidates</b> to accelerate triage and expert review.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

# ---- Footer ----
st.markdown("""
<div class='mx' style='text-align:center; color:#ffffff; padding: 1.6rem 0; border-top: 1px solid rgba(102,126,234,.35); margin-top:.6rem;'>
  <div style='font-size:.95rem; opacity:.9;'>ExoMatch · Professional Exoplanet Vetting Models</div>
  <div style='font-size:.85rem; opacity:.65; margin-top:.35rem;'>Design language aligned with Analysis/Vetting pages · Build 2025</div>
</div>
""", unsafe_allow_html=True)
