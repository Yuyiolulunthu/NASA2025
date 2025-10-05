import streamlit as st
import numpy as np
import pandas as pd
from astropy.io import fits
from io import BytesIO
import sys
import os
from datetime import datetime

# Fix import path for local components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from components.banner import render_banner
    has_banner = True
except:
    has_banner = False

# ---------- Page Config ----------
st.set_page_config(
    page_title="FITS to CSV Converter - ExoMatch",
    page_icon="Web/logo.png",
    layout="wide",
)

if has_banner:
    render_banner()

st.markdown("""
    <style>
    #MainMenu, footer, header {visibility:hidden;}
    </style>
""", unsafe_allow_html=True)

# ---------- CSS ----------
st.markdown("""
<style>
/* ===== Globals ===== */
[data-testid="stAppViewContainer"]{
  background: radial-gradient(circle at 50% 50%, #0b0f1a 0%, #0e162a 60%, #0a0e27 100%);
  color:#eaeef7; font-family:'Inter','Open Sans',sans-serif; overflow-x:hidden;
}

/* ===== Headings ===== */
h1.app-title{
  text-align:center; 
  color:#eaf2ff; 
  font-size:2.2rem; 
  letter-spacing:.5px;
  margin: 0 0 0.5rem 0;
  font-weight:800;
}
p.app-subtitle{
  text-align:center; 
  color:#a9b7d9; 
  margin-top:0;
}

/* ===== Button Styles ===== */
.stButton > button {
  background: rgba(20, 30, 60, 0.85) !important;
  color: #ffffff !important;
  border: 2px solid rgba(80, 120, 200, 0.6) !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-size: 1.05rem !important;
  padding: 0.75rem 1.5rem !important;
  transition: all 0.3s ease !important;
}
.stButton > button:hover {
  background: rgba(63, 169, 245, 0.25) !important;
  border-color: rgba(63, 169, 245, 0.9) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 4px 12px rgba(63, 169, 245, 0.3) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }

/* ===== Download Button Styles ===== */
.stDownloadButton > button {
  background: rgba(63, 169, 245, 0.3) !important;
  color: #ffffff !important;
  border: 2px solid rgba(63, 169, 245, 0.8) !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-size: 1.05rem !important;
  padding: 0.75rem 1.5rem !important;
  transition: all 0.3s ease !important;
}
.stDownloadButton > button:hover {
  background: rgba(63, 169, 245, 0.5) !important;
  border-color: rgba(63, 169, 245, 1) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 4px 16px rgba(63, 169, 245, 0.5) !important;
}

/* ===== Cards ===== */
.info-card{
  background:rgba(20,30,60,.72); 
  border:1px solid rgba(80,120,200,.42);
  border-radius:16px; 
  padding:1.5rem; 
  margin: 1rem 0;
  box-shadow:0 4px 20px rgba(0,0,0,.3);
}
.info-card h3 { color:#3fa9f5; margin:0 0 .5rem 0; }

/* ===== File Upload Area ===== */
[data-testid="stFileUploader"] {
  background: rgba(20,30,60,.5);
  border: 2px dashed rgba(80, 120, 200, 0.5);
  border-radius: 12px;
  padding: 2rem;
}
[data-testid="stFileUploader"] label {
  color: #ffffff !important;
  font-size: 1.1rem !important;
  font-weight: 600 !important;
}

/* ===== Expander ===== */
[data-testid="stExpander"] {
  background: rgba(20,30,60,.5);
  border: 1px solid rgba(80, 120, 200, 0.3);
  border-radius: 8px;
}

/* ===== Dataframe ===== */
[data-testid="stDataFrame"] { background: rgba(20,30,60,.7); }

/* ===== Tabs ===== */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
  background-color: rgba(20, 30, 60, 0.5);
  border-radius: 8px 8px 0 0;
  color: #ffffff;
  padding: 0.5rem 1rem;
}
.stTabs [aria-selected="true"]{
  background-color: rgba(63, 169, 245, 0.3);
  border-bottom: 2px solid #3fa9f5;
}
</style>
""", unsafe_allow_html=True)

# ---------- Helper Functions ----------
def extract_time_from_header(header):
    """Extract observation time from FITS header"""
    time_keywords = ['DATE-OBS', 'MJD-OBS', 'JD', 'TIME-OBS', 'OBSTIME']
    for keyword in time_keywords:
        if keyword in header:
            return str(header[keyword])
    return None

def simple_aperture_photometry(data, x=None, y=None, radius=10):
    """Simple aperture photometry without photutils dependency"""
    if data is None or len(data.shape) != 2:
        return None
    # use center if not provided
    if x is None or y is None:
        y, x = data.shape[0] // 2, data.shape[1] // 2
    yy, xx = np.ogrid[:data.shape[0], :data.shape[1]]
    mask = (xx - x)**2 + (yy - y)**2 <= radius**2
    flux = np.sum(data[mask])
    # background annulus
    bg_inner = radius + 2
    bg_outer = radius + 8
    bg_mask = ((xx - x)**2 + (yy - y)**2 > bg_inner**2) & ((xx - x)**2 + (yy - y)**2 <= bg_outer**2)
    if np.sum(bg_mask) > 0:
        bg_median = np.median(data[bg_mask])
        n_pixels = np.sum(mask)
        flux_corrected = flux - (bg_median * n_pixels)
    else:
        flux_corrected = flux
    return {'flux': flux_corrected, 'x': x, 'y': y, 'radius': radius}

# ---------- Header ----------
st.markdown("<h1 class='app-title'>FITS to CSV Converter</h1>", unsafe_allow_html=True)
st.markdown("<p class='app-subtitle'>ExoMatch · Professional Light-Curve Workflow</p>", unsafe_allow_html=True)

# ---------- Highlights (English only) ----------
with st.container():
    st.markdown("""
    <div class="info-card">
      <h3>Highlights</h3>
      <ul style="margin-top:.5rem; line-height:1.8;">
        <li><b>ExoClock-compatible workflow:</b> Integrates the photometry and light-curve plotting logic aligned with ESA/ExoClock needs so you can produce a ready-to-use CSV here.</li>
        <li><b>All-in-one analysis:</b> Single-file inspection, time-series photometry, light-curve preview, model evaluation, and manual calibration (target position and aperture) in one place.</li>
        <li><b>Focused UI:</b> A clean dark theme designed for long vetting sessions and repeated comparisons.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["Single File Conversion", "Time Series (Multiple Files)"])

# ========== TAB 1: Single File Conversion ==========
with tab1:
    st.markdown("""
    <div class="info-card">
      <h3>Single File Converter</h3>
      <p style="font-size:1.05rem; color:#ffffff; line-height:1.7;">
        Analyze a single FITS file and export either full image data, row-wise statistics, or FITS header information. A quick way to visualize and inspect one dataset.
      </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Select FITS File", type=['fits', 'fit', 'fts'], key="single_file")

    if uploaded_file is not None:
        try:
            fits_data = fits.open(uploaded_file)
            st.info(f"Successfully loaded: {uploaded_file.name}")

            # File info
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="info-card"><h3>Basic Info</h3>', unsafe_allow_html=True)
                st.write(f"**Filename:** {uploaded_file.name}")
                st.write(f"**HDUs:** {len(fits_data)}")
                st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="info-card"><h3>Image Info</h3>', unsafe_allow_html=True)
                primary_hdu = fits_data[0]
                if primary_hdu.data is not None:
                    st.write(f"**Shape:** {primary_hdu.data.shape}")
                    st.write(f"**Type:** {primary_hdu.data.dtype}")
                    st.write(f"**Pixels:** {primary_hdu.data.size:,}")
                else:
                    st.warning("No image data in primary HDU")
                st.markdown('</div>', unsafe_allow_html=True)

            conversion_mode = st.radio(
                "Conversion Mode",
                ["Full Image Data", "Statistical Summary", "Header Information"],
                horizontal=True
            )

            if st.button("Convert to CSV", key="convert_single"):
                if primary_hdu.data is None and conversion_mode != "Header Information":
                    st.error("No data in selected HDU.")
                else:
                    with st.spinner("Converting..."):
                        if conversion_mode == "Full Image Data":
                            if len(primary_hdu.data.shape) == 2:
                                df = pd.DataFrame(primary_hdu.data)
                                df.columns = [f"Col_{i}" for i in range(df.shape[1])]
                                df.insert(0, 'Row', range(len(df)))
                            else:
                                flat_data = primary_hdu.data.flatten()
                                df = pd.DataFrame({'Index': range(len(flat_data)), 'Value': flat_data})

                        elif conversion_mode == "Statistical Summary":
                            if len(primary_hdu.data.shape) >= 2:
                                stats_data = []
                                for i, row in enumerate(primary_hdu.data):
                                    stats_data.append({
                                        'Row': i,
                                        'Mean': np.mean(row),
                                        'Median': np.median(row),
                                        'Std': np.std(row),
                                        'Min': np.min(row),
                                        'Max': np.max(row)
                                    })
                                df = pd.DataFrame(stats_data)
                            else:
                                st.error("Requires 2D data")
                                df = None

                        else:  # Header Information
                            df = pd.DataFrame({
                                'Keyword': list(primary_hdu.header.keys()),
                                'Value': [str(v) for v in primary_hdu.header.values()],
                                'Comment': list(primary_hdu.header.comments)
                            })

                        if df is not None:
                            st.dataframe(df.head(50), use_container_width=True, height=300)
                            csv_buffer = BytesIO()
                            df.to_csv(csv_buffer, index=False, encoding='utf-8')
                            st.download_button(
                                label="Download CSV",
                                data=csv_buffer.getvalue(),
                                file_name=uploaded_file.name.rsplit('.', 1)[0] + '.csv',
                                mime='text/csv',
                                use_container_width=True
                            )

            fits_data.close()

        except Exception as e:
            st.error(f"Error: {str(e)}")

# ========== TAB 2: Time Series ==========
with tab2:
    st.markdown("""
    <div class="info-card">
      <h3>Time Series Analyzer</h3>
      <p style="font-size:1.05rem; color:#ffffff; line-height:1.7;">
        Build a light curve from multiple FITS frames. Supports auto-detection of the brightest target or manual coordinates, and adjustable aperture radius for different seeing/fields. Export results as a CSV for downstream analysis or ExoClock submission.
      </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Select Multiple FITS Files",
        type=['fits', 'fit', 'fts'],
        accept_multiple_files=True,
        key="multiple_files"
    )

    if uploaded_files:
        st.info(f"Loaded {len(uploaded_files)} files")

        # Photometry settings
        st.markdown("### Photometry Settings")
        col1, col2, col3 = st.columns(3)
        with col1:
            use_auto_center = st.checkbox("Auto-detect star position", value=True)
        with col2:
            aperture_radius = st.slider("Aperture radius (pixels)", 5, 30, 10)
        with col3:
            sort_by_time = st.checkbox("Sort by observation time", value=True)

        if not use_auto_center:
            col_x, col_y = st.columns(2)
            with col_x:
                manual_x = st.number_input("Star X position", min_value=0, value=100)
            with col_y:
                manual_y = st.number_input("Star Y position", min_value=0, value=100)

        if st.button("Generate Time Series", use_container_width=True):
            with st.spinner(f"Processing {len(uploaded_files)} files..."):
                time_series_data = []
                progress_bar = st.progress(0)

                for idx, file in enumerate(uploaded_files):
                    try:
                        hdul = fits.open(file)
                        header = hdul[0].header
                        data = hdul[0].data

                        if data is not None:
                            # Extract time
                            obs_time = extract_time_from_header(header)

                            # Auto-center by brightest pixel or use manual/default center
                            if use_auto_center and len(data.shape) == 2:
                                y_max, x_max = np.unravel_index(np.argmax(data), data.shape)
                                x_center, y_center = x_max, y_max
                            else:
                                x_center = manual_x if not use_auto_center else data.shape[1] // 2
                                y_center = manual_y if not use_auto_center else data.shape[0] // 2

                            # Photometry
                            phot_result = simple_aperture_photometry(
                                data, x=x_center, y=y_center, radius=aperture_radius
                            )

                            if phot_result:
                                time_series_data.append({
                                    'Filename': file.name,
                                    'Observation_Time': obs_time if obs_time else f"Frame_{idx:04d}",
                                    'Flux': phot_result['flux'],
                                    'X_Position': phot_result['x'],
                                    'Y_Position': phot_result['y'],
                                    'File_Index': idx
                                })

                        hdul.close()

                    except Exception as e:
                        st.warning(f"Error processing {file.name}: {str(e)}")

                    progress_bar.progress((idx + 1) / len(uploaded_files))

                if time_series_data:
                    df = pd.DataFrame(time_series_data)

                    # Sort by time if requested
                    if sort_by_time and 'Observation_Time' in df.columns:
                        try:
                            df = df.sort_values('Observation_Time')
                        except:
                            df = df.sort_values('File_Index')

                    # Normalize flux to relative brightness
                    if len(df) > 0:
                        median_flux = df['Flux'].median()
                        df['Relative_Brightness'] = df['Flux'] / median_flux

                    st.success(f"Processed {len(df)} frames.")

                    # Display results
                    st.markdown("### Light Curve Preview")
                    st.line_chart(df.set_index('File_Index')['Relative_Brightness'])

                    st.markdown("### Data Preview")
                    st.dataframe(df, use_container_width=True, height=300)

                    # Download
                    st.markdown("### Download Time Series")
                    csv_buffer = BytesIO()
                    df.to_csv(csv_buffer, index=False, encoding='utf-8')
                    st.download_button(
                        label="Download Light Curve CSV",
                        data=csv_buffer.getvalue(),
                        file_name=f"lightcurve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv',
                        use_container_width=True
                    )

                    st.info(f"The CSV contains {len(df)} time points with flux measurements.")
                else:
                    st.error("No valid data extracted from files.")

# ---------- Usage Guide (English only) ----------
with st.expander("Usage Guide"):
    st.markdown("""
    **Single File Conversion**  
    - Upload one FITS file.  
    - Choose one of the export modes:  
      - Full Image Data: full pixel matrix with row/column indices.  
      - Statistical Summary: per-row Mean / Median / Std / Min / Max.  
      - Header Information: FITS header (Keyword / Value / Comment).  

    **Time Series (Multiple Files)**  
    - Upload multiple FITS frames from the same observation sequence.  
    - Auto-detect the brightest target or set manual coordinates.  
    - Adjust the aperture radius for your field and seeing.  
    - Preview the light curve and download a CSV for analysis or ExoClock submission.  

    **Notes**  
    - Use calibrated images (Bias/Dark/Flat) for best results.  
    - Files should cover the same field of view.  
    - Output can be used for transit analysis, model fitting, and manual calibration.
    """)

st.markdown("<br><br><div style='text-align:center;color:#aaa;'>ExoMatch — FITS Converter & Light-Curve Workflow</div>", unsafe_allow_html=True)
