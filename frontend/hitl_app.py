"""
🌌 深色太空主題 + 卡片滑動風格
AI-人類協作式系外行星辨識平台
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# 頁面配置
st.set_page_config(
    page_title="🌌 Exoplanet Hunter",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 深色太空主題 CSS
st.markdown("""
<style>
    /* 整體背景 - 深色太空 */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #16213e 50%, #0f3460 100%);
        color: #e0e0e0;
    }
    
    /* 星空背景動畫 */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, #eee, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 60px 70px, #fff, rgba(0,0,0,0)),
            radial-gradient(1px 1px at 50px 50px, #fff, rgba(0,0,0,0)),
            radial-gradient(1px 1px at 130px 80px, #fff, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 90px 10px, #fff, rgba(0,0,0,0));
        background-repeat: repeat;
        background-size: 200px 200px;
        opacity: 0.4;
        z-index: -1;
        animation: twinkle 3s ease-in-out infinite;
    }
    
    @keyframes twinkle {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 0.6; }
    }
    
    /* 主標題 - 發光效果 */
    .space-title {
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 3s ease infinite;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
        padding: 2rem 0;
        letter-spacing: 0.1em;
    }
    
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* 卡片樣式 - 太空感 */
    .candidate-card {
        background: linear-gradient(135deg, rgba(22, 33, 62, 0.95), rgba(15, 52, 96, 0.95));
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 
            0 8px 32px 0 rgba(31, 38, 135, 0.37),
            inset 0 0 20px rgba(102, 126, 234, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .candidate-card::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.1) 0%, transparent 70%);
        animation: rotate-gradient 10s linear infinite;
    }
    
    @keyframes rotate-gradient {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .candidate-card:hover {
        border-color: rgba(102, 126, 234, 0.8);
        box-shadow: 
            0 12px 40px 0 rgba(31, 38, 135, 0.5),
            inset 0 0 30px rgba(102, 126, 234, 0.2),
            0 0 40px rgba(102, 126, 234, 0.4);
        transform: translateY(-5px);
    }
    
    /* 按鈕 - 發光效果 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 1rem 2rem;
        font-size: 1.2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* 特殊按鈕顏色 */
    div[data-testid="column"]:nth-child(1) .stButton > button {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4);
    }
    
    div[data-testid="column"]:nth-child(3) .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }
    
    /* 指標卡片 */
    .metric-card {
        background: rgba(22, 33, 62, 0.6);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* 進度條 */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.6);
    }
    
    /* 文字發光 */
    .glow-text {
        color: #667eea;
        text-shadow: 0 0 10px rgba(102, 126, 234, 0.8);
    }
    
    /* 隱藏 Streamlit 預設元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 側邊欄深色 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e27 0%, #16213e 100%);
    }
    
    /* 卡片滑動提示動畫 */
    @keyframes swipe-hint {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-10px); }
        75% { transform: translateX(10px); }
    }
    
    .swipe-hint {
        animation: swipe-hint 2s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)

# 初始化 Session State
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'candidate_index' not in st.session_state:
    st.session_state.candidate_index = 0
if 'candidates' not in st.session_state:
    # 生成測試候選
    st.session_state.candidates = []
    for i in range(10):
        time_data = np.linspace(0, 50, 500)
        flux = 1 + np.random.normal(0, 0.002, 500)
        period = np.random.uniform(5, 30)
        for t in np.arange(0, 50, period):
            mask = (time_data > t-1) & (time_data < t+1)
            flux[mask] -= np.random.uniform(0.005, 0.02)
        
        st.session_state.candidates.append({
            'id': f'KOI-{2000+i}',
            'time': time_data,
            'flux': flux,
            'period': period,
            'depth': np.random.uniform(0.5, 2.0),
            'duration': np.random.uniform(2, 6),
            'snr': np.random.uniform(10, 50),
            'ai_confidence': np.random.uniform(0.5, 0.95)
        })
if 'labels' not in st.session_state:
    st.session_state.labels = []

# 工具函數
def create_lightcurve_plot(candidate):
    """創建光變曲線圖表 - 深色主題"""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.1,
        subplot_titles=('完整光變曲線', '凌日特寫')
    )
    
    # 完整曲線
    fig.add_trace(
        go.Scatter(
            x=candidate['time'],
            y=candidate['flux'],
            mode='lines',
            name='光變曲線',
            line=dict(color='#4facfe', width=2),
            fill='tozeroy',
            fillcolor='rgba(79, 172, 254, 0.1)'
        ),
        row=1, col=1
    )
    
    # 標註凌日區域
    for t in np.arange(0, 50, candidate['period']):
        fig.add_vrect(
            x0=t-1, x1=t+1,
            fillcolor="rgba(239, 68, 68, 0.3)",
            layer="below",
            line_width=0,
            row=1, col=1
        )
    
    # 凌日特寫
    transit_mask = (candidate['time'] > 0) & (candidate['time'] < candidate['period']*2)
    fig.add_trace(
        go.Scatter(
            x=candidate['time'][transit_mask],
            y=candidate['flux'][transit_mask],
            mode='markers+lines',
            name='凌日細節',
            line=dict(color='#f093fb', width=3),
            marker=dict(size=6, color='#f093fb', symbol='circle')
        ),
        row=2, col=1
    )
    
    # 深色主題設定
    fig.update_layout(
        height=600,
        paper_bgcolor='rgba(10, 14, 39, 0.8)',
        plot_bgcolor='rgba(22, 33, 62, 0.6)',
        font=dict(color='#e0e0e0', size=12),
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    fig.update_xaxes(
        gridcolor='rgba(102, 126, 234, 0.2)',
        title_text="時間 (天)",
        title_font=dict(color='#667eea')
    )
    fig.update_yaxes(
        gridcolor='rgba(102, 126, 234, 0.2)',
        title_text="相對流量",
        title_font=dict(color='#667eea')
    )
    
    return fig

# ==================== 頁面：首頁 ====================
if st.session_state.page == 'home':
    
    # 主標題
    st.markdown('<h1 class="space-title">🌌 EXOPLANET HUNTER</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.5rem; color: #a0a0a0; letter-spacing: 0.2em;">AI × HUMAN COLLABORATION PLATFORM</p>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 使命卡片
    st.markdown("""
    <div class="candidate-card">
        <h2 style='text-align: center; color: #667eea; text-shadow: 0 0 20px rgba(102, 126, 234, 0.6);'>
            ✨ OUR MISSION ✨
        </h2>
        <p style='text-align: center; font-size: 1.3rem; line-height: 1.8; margin-top: 1rem;'>
            結合 <span class="glow-text"><strong>AI 的極速分析</strong></span> 與 
            <span class="glow-text"><strong>人類的直覺智慧</strong></span><br>
            共同探索宇宙中的 <span style="color: #f093fb; font-weight: bold;">新世界</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 三大特色
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style='font-size: 3rem; margin-bottom: 0.5rem;'>🤖</div>
            <h3 style='color: #667eea;'>AI 初篩</h3>
            <p style='color: #a0a0a0; margin-top: 0.5rem;'>
                自動分析數千筆資料<br>
                快速識別潛在信號<br>
                標註不確定案例
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style='font-size: 3rem; margin-bottom: 0.5rem;'>👁️</div>
            <h3 style='color: #4facfe;'>人類判斷</h3>
            <p style='color: #a0a0a0; margin-top: 0.5rem;'>
                審查邊界案例<br>
                滑動操作直覺<br>
                你的智慧關鍵
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style='font-size: 3rem; margin-bottom: 0.5rem;'>🔄</div>
            <h3 style='color: #f093fb;'>協作學習</h3>
            <p style='color: #a0a0a0; margin-top: 0.5rem;'>
                標註改進模型<br>
                追蹤貢獻度<br>
                持續進化升級
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # 開始按鈕
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 START EXPLORATION", use_container_width=True, type="primary"):
            st.session_state.page = 'review'
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 統計資訊
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='text-align: center;'>
            <div style='font-size: 2.5rem; color: #667eea; font-weight: bold;'>{}</div>
            <div style='color: #a0a0a0;'>你的標註</div>
        </div>
        """.format(len(st.session_state.labels)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center;'>
            <div style='font-size: 2.5rem; color: #4facfe; font-weight: bold;'>{}</div>
            <div style='color: #a0a0a0;'>待審查</div>
        </div>
        """.format(len(st.session_state.candidates) - st.session_state.candidate_index), unsafe_allow_html=True)
    
    with col3:
        confirmed = len([l for l in st.session_state.labels if l == 'CONFIRMED'])
        st.markdown("""
        <div style='text-align: center;'>
            <div style='font-size: 2.5rem; color: #10b981; font-weight: bold;'>{}</div>
            <div style='color: #a0a0a0;'>確認行星</div>
        </div>
        """.format(confirmed), unsafe_allow_html=True)
    
    with col4:
        accuracy = np.random.uniform(80, 95) if len(st.session_state.labels) > 0 else 0
        st.markdown("""
        <div style='text-align: center;'>
            <div style='font-size: 2.5rem; color: #f093fb; font-weight: bold;'>{:.1f}%</div>
            <div style='color: #a0a0a0;'>準確度</div>
        </div>
        """.format(accuracy), unsafe_allow_html=True)

# ==================== 頁面：卡片審查 ====================
elif st.session_state.page == 'review':
    
    candidates = st.session_state.candidates
    idx = st.session_state.candidate_index
    
    if idx >= len(candidates):
        st.markdown('<h1 class="space-title">🎉 ALL DONE!</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div class="candidate-card" style="text-align: center;">
            <h2 style='color: #667eea;'>你已審查完所有候選！</h2>
            <p style='font-size: 1.3rem; margin: 2rem 0;'>
                感謝你的貢獻！<br>
                你幫助 AI 變得更聰明了 🚀
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🏠 返回首頁", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
        return
    
    current = candidates[idx]
    
    # 頂部進度
    progress = (idx + 1) / len(candidates)
    st.progress(progress)
    st.markdown(f"""
    <div style='text-align: center; margin: 1rem 0;'>
        <span style='color: #667eea; font-size: 1.2rem; font-weight: bold;'>
            CANDIDATE {idx + 1} / {len(candidates)}
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # 候選卡片
    st.markdown(f"""
    <div class="candidate-card">
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;'>
            <h2 style='color: #667eea; margin: 0;'>🪐 {current['id']}</h2>
            <div style='text-align: right;'>
                <div style='font-size: 0.9rem; color: #a0a0a0;'>AI 信心度</div>
                <div style='font-size: 2rem; color: #f093fb; font-weight: bold;'>{current['ai_confidence']:.0%}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 光變曲線圖表
    fig = create_lightcurve_plot(current)
    st.plotly_chart(fig, use_container_width=True)
    
    # 候選資訊卡片
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("軌道週期", f"{current['period']:.2f} 天", col1),
        ("凌日深度", f"{current['depth']:.2f}%", col2),
        ("持續時間", f"{current['duration']:.2f} 小時", col3),
        ("信噪比", f"{current['snr']:.1f}", col4)
    ]
    
    for label, value, col in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style='color: #a0a0a0; font-size: 0.9rem;'>{label}</div>
                <div style='color: #667eea; font-size: 1.5rem; font-weight: bold; margin-top: 0.5rem;'>{value}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 滑動提示
    st.markdown("""
    <div style='text-align: center; margin: 2rem 0;'>
        <p class='swipe-hint' style='color: #667eea; font-size: 1.2rem;'>
            👈 滑動判斷 👉
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 滑動操作按鈕
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("👈 假陽性\nFALSE POSITIVE", use_container_width=True, key="fp"):
            st.session_state.labels.append('FALSE_POSITIVE')
            st.session_state.candidate_index += 1
            st.success("✅ 已標註為假陽性")
            time.sleep(0.5)
            st.rerun()
    
    with col2:
        if st.button("👉 候選行星\nCANDIDATE", use_container_width=True, key="candidate"):
            st.session_state.labels.append('CANDIDATE')
            st.session_state.candidate_index += 1
            st.info("✅ 已標註為候選")
            time.sleep(0.5)
            st.rerun()
    
    with col3:
        if st.button("👆 確認行星\nCONFIRMED", use_container_width=True, key="confirmed"):
            st.session_state.labels.append('CONFIRMED')
            st.session_state.candidate_index += 1
            st.balloons()
            st.success("✅ 已標註為確認行星！")
            time.sleep(0.5)
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 跳過和返回
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⏭️ 跳過", use_container_width=True):
            st.session_state.candidate_index += 1
            st.rerun()
    with col2:
        if st.button("🏠 返回首頁", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()

# 頁腳
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #667eea; padding: 2rem;'>
    <p style='font-size: 0.9rem; opacity: 0.6;'>
        🌌 EXOPLANET HUNTER v2.0 | Powered by AI × Human Intelligence
    </p>
</div>
""", unsafe_allow_html=True)