# Web/pages/about.py
import streamlit as st
from components.banner import render_banner

# Try to render the global top banner if your project has it.
try:
    from components.banner import render_banner
except Exception:
    render_banner = None

# ---------------- Page setup ----------------
st.set_page_config(
    page_title="About Our Models — ExoMatch",
    page_icon="Web/logo.png",
    layout="wide",
    initial_sidebar_state="collapsed",
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

if render_banner:
    render_banner()

# ---------------- Minimal professional styling ----------------
st.markdown("""
<style>
  .stApp {
    background: linear-gradient(135deg, #0a0e27 0%, #16213e 50%, #0f3460 100%);
    color: #e0e0e0;
  }
  .content {
    max-width: 1100px;
    margin: 0 auto;
  }
  .eyebrow {
    text-transform: uppercase;
    letter-spacing: .12em;
    font-weight: 700;
    color: #9aa7ff;
    font-size: .85rem;
  }
  h1, h2, h3 {
    letter-spacing: .02em;
  }
  h1 {
    font-weight: 900;
    background: linear-gradient(45deg, #667eea, #4facfe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: .25rem;
  }
  .lead {
    color: #cad4ff;
    font-size: 1.1rem;
    opacity: .95;
  }
  .card {
    background: rgba(22, 33, 62, 0.60);
    border: 1px solid rgba(102, 126, 234, 0.30);
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 1rem;
  }
  .muted { color: #cbd5e1; }
  .callout {
    border-left: 4px solid #667eea;
    padding-left: 1rem;
  }
  table { width: 100%; }
  thead th {
    background: rgba(102,126,234,0.14);
    border-bottom: 1px solid rgba(102,126,234,0.35);
    color: #e5e7eb !important;
  }
  td, th {
    padding: .6rem .5rem !important;
  }
  .small { font-size: .95rem; }
  .kpi {
    display: inline-block;
    background: rgba(22, 33, 62, 0.75);
    border: 1px solid rgba(102,126,234,.3);
    padding: .35rem .6rem;
    border-radius: 8px;
    margin-right: .35rem;
    margin-bottom: .35rem;
    font-weight: 700;
    color: #b9c3ff;
  }
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown("""
<div class="content">
  <div class="eyebrow">ExoMatch</div>
  <h1>About Our Models</h1>
  <p class="lead">We build a lightweight, explainable machine-learning stack for exoplanet vetting, drawing on
  published research and adapting it for professional workflows that balance accuracy, speed, and interpretability.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='content'>", unsafe_allow_html=True)

# ---------------- Papers section ----------------
st.markdown("### Papers We Draw Upon")

with st.container():
    st.markdown("#### Exoplanet Detection Using Machine Learning")
    st.markdown("""
<div class="card small">
  <div class="callout">
    <p class="muted">Traditional transit searches (e.g., Box-Least Squares, BLS) deteriorate under low S/N or heavy systematics.
    Pure deep networks may boost accuracy but at a higher computational cost and with limited interpretability.
    We therefore reference a lightweight ML framework that preserves performance while improving efficiency and explainability.</p>
  </div>
  <ol>
    <li><strong>Pre-processing</strong> — Clean raw light curves: remove systematics, impute gaps, and de-trend to stabilize inputs.</li>
    <li><strong>Feature extraction</strong> — Compute <em>~789 time-series descriptors</em> (variance, energy, peaks, etc.) via automated libraries (e.g., <code>tsfresh</code>), minimizing manual feature bias.</li>
    <li><strong>Classifier</strong> — Gradient-boosted decision trees (e.g., <strong>LightGBM</strong>) to predict “planet candidate” vs. non-planet.</li>
    <li><strong>Evaluation data</strong> — Train/validate on <strong>Kepler</strong>; test on <strong>TESS</strong>; stress test with simulated K2 transits.</li>
  </ol>
  <p><strong>Reported performance</strong></p>
  <div>
    <span class="kpi">Kepler AUC ≈ 0.948</span>
    <span class="kpi">Kepler Recall ≈ 96%</span>
    <span class="kpi">TESS Accuracy ≈ 98%</span>
    <span class="kpi">TESS Recall ≈ 82%</span>
    <span class="kpi">TESS Precision ≈ 63%</span>
  </div>
  <p><strong>Advantages</strong> — CPU-friendly (minutes to train), interpretable feature importances, telescope-agnostic (Kepler/K2/TESS), and no mandatory phase-folding pipeline.</p>
  <p><strong>Limitations</strong> — Slightly below state-of-the-art deep nets; imbalanced TESS labels depress precision; still requires expert vetting; boosted trees are not fully transparent.</p>
  <p><strong>Outlook</strong> — Multi-mission global models, imbalance remedies (augmentation/semi-supervised), and unsupervised structure discovery for novel signal types.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("#### Assessment of Ensemble-Based ML Algorithms for Exoplanet Identification")
    st.markdown("""
<div class="card small">
  <div class="callout">
    <p class="muted">We also reference ensemble strategies—bagging, boosting, and stacking—evaluated on Kepler light curves
    to improve robustness under noise and reduce false positives compared to single models or BLS-style pipelines.</p>
  </div>
  <ol>
    <li><strong>Data</strong> — Kepler LCs, preprocessed (de-trending, normalization) with positive (planet) and negative examples.</li>
    <li><strong>Features</strong> — Time-series statistics capturing amplitude, period, depth, energy, and distributional shape.</li>
    <li><strong>Models</strong> — Single learners (RF, GBM, XGBoost, SVM, Decision Trees) vs. <strong>ensembles</strong> (Voting, Bagging, Boosting, <strong>Stacking</strong>).</li>
  </ol>
  <p><strong>Reported performance</strong></p>
  <div>
    <span class="kpi">Accuracy (single): 95–97%</span>
    <span class="kpi">Accuracy (stacking): ~98.5%</span>
    <span class="kpi">Recall (single): 80–85%</span>
    <span class="kpi">Recall (ensemble): &gt;90%</span>
    <span class="kpi">Precision (single): 75–80%</span>
    <span class="kpi">Precision (ensemble): 85–88%</span>
    <span class="kpi">F1 (single): ~0.80–0.85</span>
    <span class="kpi">F1 (ensemble): ~0.91–0.93</span>
    <span class="kpi">ROC-AUC (ensemble): ~0.97–0.98</span>
  </div>
  <p><strong>Key findings</strong> — Ensembles, especially stacking, consistently outperform single models in low S/N regimes; period, depth, and symmetry dominate feature importance; false positives are reduced relative to BLS.</p>
  <p><strong>Limitations</strong> — Higher training cost than single learners; class imbalance still degrades performance in edge cases.</p>
  <p><strong>Conclusion</strong> — Ensemble learning offers a strong balance of precision/recall and generalization, bridging classical methods and deep learning.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Side-by-side comparison ----------------
st.markdown("#### Comparative Summary")
st.markdown("""
<table>
  <thead>
    <tr>
      <th>Aspect</th>
      <th>“Exoplanet Detection using ML”</th>
      <th>“Assessment of Ensemble-Based ML Algorithms”</th>
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
      <td>Automated features (~789 via <code>tsfresh</code>); classifier: <strong>LightGBM</strong></td>
      <td>Multiple single models vs. <strong>ensembles</strong> (Voting/Bagging/Boosting/Stacking)</td>
    </tr>
    <tr>
      <td>Compute</td>
      <td>Lightweight; CPU minutes; no GPU required</td>
      <td>Higher cost than single models; still lighter than deep nets</td>
    </tr>
    <tr>
      <td>Kepler Results</td>
      <td>AUC ≈ 0.948; Recall ≈ 96%</td>
      <td>Accuracy ≈ 98.5% (stacking); Recall &gt; 90%; Precision ≈ 85–88%; F1 ≈ 0.91–0.93; ROC-AUC ≈ 0.97–0.98</td>
    </tr>
    <tr>
      <td>Strengths</td>
      <td>Quick to train; interpretable; mission-agnostic</td>
      <td>Higher overall accuracy; better precision/recall tradeoff; robust generalization</td>
    </tr>
    <tr>
      <td>Limitations</td>
      <td>Lower precision on imbalanced TESS; not fully transparent</td>
      <td>More training time; class imbalance remains challenging</td>
    </tr>
  </tbody>
</table>
""", unsafe_allow_html=True)

# ---------------- Our approach: fused workflow ----------------
st.markdown("### Our Approach: A Pragmatic, Explainable Ensemble Pipeline")
st.markdown("""
<div class="card">
  <ol>
    <li><strong>Feature extraction</strong> — Use <code>tsfresh</code> (or <code>sktime</code>) to compute hundreds of robust descriptors from raw light curves.
        Augment with BLS/TLS-derived period, depth, and duration as structured features.</li>
    <li><strong>Base learners</strong> — Train LightGBM/XGBoost/CatBoost plus Random Forest/ExtraTrees (tree baselines) and a linear baseline (Logistic Regression or SVM).</li>
    <li><strong>Ensembling</strong> — Start with weighted voting; then adopt <strong>stacking</strong> with a simple meta-learner
        (Logistic Regression/LightGBM) on out-of-fold probabilities.</li>
    <li><strong>Evaluation</strong> — K-fold CV with ROC-AUC, PR-AUC, F1, and <em>Recall @ 1% FPR</em> to benchmark against BLS-style baselines.</li>
  </ol>
  <p class="muted"><em>Why this blend?</em> It keeps training/cycle times short (CPU-friendly), exposes feature importance for QA, and
  gains robustness from multiple learners—well-suited for production vetting and human-in-the-loop review.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Advantages & Caveats ----------------
cols = st.columns(2)
with cols[0]:
    st.markdown("#### Advantages")
    st.markdown("""
- **Fast iteration** — No GPU requirement; minutes per retrain on CPUs.
- **Explainability** — Global and per-sample feature importances facilitate scientific review.
- **Generalization** — Trained across missions; resilient to instrument differences after pre-processing.
- **Operational fit** — Works seamlessly with human vetting (ranked candidates + uncertainty flags).
""")
with cols[1]:
    st.markdown("#### Caveats")
    st.markdown("""
- **Below best deep nets in extreme low S/N** — Ensembles narrow the gap but do not eliminate it.
- **Class imbalance sensitivity** — Use <code>class_weight=balanced</code> / <code>scale_pos_weight</code>, calibrated thresholds, and PR-AUC monitoring.
- **Residual opacity** — Tree ensembles are more interpretable than deep nets, but not fully transparent.
""")

# ---------------- Forward look ----------------
st.markdown("### Roadmap")
st.markdown("""
- **Unified multi-mission model** with domain adaptation to reduce calibration drift across Kepler/K2/TESS.
- **Imbalance remedies**: augmentation, semi-supervised positive-unlabeled (PU) learning, and calibrated decision thresholds by target class.
- **Unsupervised discovery** (autoencoders, clustering) to surface novel transit-like anomalies.
- **Application to unconfirmed candidates** to accelerate triage and expert review.
""")

st.markdown("</div>", unsafe_allow_html=True)
