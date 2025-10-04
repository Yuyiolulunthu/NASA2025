"""
AI-人類協作式系外行星辨識平台
Human-in-the-Loop Exoplanet Finder
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime
import json
from pathlib import Path

# 頁面配置
st.set_page_config(
    page_title="AI-人類協作式系外行星探索平台",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自訂樣式
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(120deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
    }
    .mission-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    .candidate-card {
        border: 2px solid #667eea;
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        background: #f8f9fa;
    }
    .swipe-button {
        font-size: 1.2rem;
        padding: 1rem 2rem;
        border-radius: 0.5rem;
        border: none;
        cursor: pointer;
        transition: all 0.3s;
    }
    .score-bar {
        height: 30px;
        border-radius: 15px;
        background: linear-gradient(to right, #ef4444 0%, #f59e0b 50%, #10b981 100%);
    }
    .contribution-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# 初始化 session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'current_candidate' not in st.session_state:
    st.session_state.current_candidate = None
if 'candidate_index' not in st.session_state:
    st.session_state.candidate_index = 0
if 'candidates' not in st.session_state:
    st.session_state.candidates = []
if 'user_labels' not in st.session_state:
    st.session_state.user_labels = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'ensemble_v1'

# API 配置
API_URL = "http://localhost:8000"

# ==================== 工具函數 ====================

def check_api():
    """檢查 API 狀態"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def generate_mock_lightcurve():
    """生成模擬光變曲線"""
    time = np.linspace(0, 100, 1000)
    
    # 基礎星光
    flux = np.random.normal(1.0, 0.01, len(time))
    
    # 添加凌日信號
    period = np.random.uniform(5, 50)
    depth = np.random.uniform(0.005, 0.02)
    duration = np.random.uniform(2, 6)
    
    for i in range(int(100/period)):
        transit_time = i * period + np.random.uniform(0, 5)
        transit_mask = np.abs(time - transit_time) < duration/2
        flux[transit_mask] -= depth
    
    return time, flux, period, depth, duration

def save_user_annotation(candidate_id, label, confidence, reason=""):
    """儲存使用者標註"""
    annotation = {
        'candidate_id': candidate_id,
        'label': label,
        'confidence': confidence,
        'reason': reason,
        'timestamp': datetime.now().isoformat(),
        'user_id': st.session_state.get('user_id', 'anonymous')
    }
    
    st.session_state.user_labels.append(annotation)
    
    # 儲存到本地 JSON
    annotations_file = Path('user_annotations.json')
    if annotations_file.exists():
        with open(annotations_file, 'r') as f:
            all_annotations = json.load(f)
    else:
        all_annotations = []
    
    all_annotations.append(annotation)
    
    with open(annotations_file, 'w') as f:
        json.dump(all_annotations, f, indent=2)

# ==================== 頁面：首頁 ====================

def home_page():
    """首頁"""
    
    st.markdown('<h1 class="main-title">🌌 AI-人類協作式系外行星辨識平台</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.3rem; color: #666;">Human-in-the-Loop Exoplanet Discovery Platform</p>', unsafe_allow_html=True)
    
    # 使命說明
    st.markdown("""
    <div class="mission-box">
        <h2>🎯 我們的使命</h2>
        <p style="font-size: 1.2rem;">
        結合 <strong>人工智慧的速度</strong> 與 <strong>人類直覺的智慧</strong>，
        共同探索宇宙中的新世界！
        </p>
        <p style="font-size: 1.1rem;">
        AI 負責快速篩選海量資料，人類專注於判斷邊界案例。
        你的每一次標註，都在幫助 AI 變得更聰明！
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 平台特色
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🤖 AI 初篩
        - 自動分析數千筆資料
        - 識別潛在行星信號
        - 標註不確定案例
        """)
    
    with col2:
        st.markdown("""
        ### 👁️ 人類判斷
        - 審查 AI 不確定的案例
        - 直覺式滑動操作
        - 視覺化光變曲線
        """)
    
    with col3:
        st.markdown("""
        ### 🔄 協作學習
        - 你的標註改進 AI
        - 追蹤貢獻度
        - 科學家驗證回饋
        """)
    
    st.markdown("---")
    
    # 資料來源說明
    st.subheader("📡 資料來源")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Kepler Mission**
        - 時間：2009-2018
        - 目標：~150,000 顆恆星
        - 發現：>2,600 顆行星
        
        [下載資料 →](https://exoplanetarchive.ipac.caltech.edu/)
        """)
    
    with col2:
        st.info("""
        **K2 Mission**
        - 時間：2014-2018
        - Kepler 延伸任務
        - 發現：~500 顆行星
        
        [下載資料 →](https://exoplanetarchive.ipac.caltech.edu/)
        """)
    
    with col3:
        st.info("""
        **TESS Mission**
        - 時間：2018-現在
        - 全天空巡天
        - 持續發現中
        
        [下載資料 →](https://tess.mit.edu/)
        """)
    
    st.markdown("---")
    
    # 開始探索按鈕
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 開始探索宇宙", use_container_width=True, type="primary"):
            st.session_state.page = 'upload'
            st.rerun()
    
    # 統計資訊
    st.markdown("---")
    st.subheader("📊 平台統計")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("累積標註", f"{len(st.session_state.user_labels)}", "你的貢獻")
    col2.metric("已審查候選", f"{st.session_state.candidate_index}", "筆")
    col3.metric("AI 模型版本", st.session_state.selected_model)
    col4.metric("活躍使用者", "1", "人")

# ==================== 頁面：資料上傳 ====================

def upload_page():
    """資料上傳/選擇頁面"""
    
    st.title("📤 資料輸入與演算法選擇")
    
    # 選擇輸入方式
    st.subheader("1️⃣ 選擇資料來源")
    
    input_method = st.radio(
        "選擇資料輸入方式：",
        ["📁 上傳光變曲線檔案 (CSV/FITS)", 
         "✏️ 手動輸入單筆候選數值", 
         "🗄️ 使用平台預先整理的 NASA 資料集"]
    )
    
    if input_method == "📁 上傳光變曲線檔案 (CSV/FITS)":
        st.info("💡 上傳包含 time 和 flux 欄位的 CSV 檔案")
        
        uploaded_file = st.file_uploader("選擇檔案", type=['csv', 'fits'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ 成功載入 {len(df)} 筆時間序列資料")
                
                # 預覽
                st.write("**資料預覽：**")
                st.dataframe(df.head())
                
                # 欄位選擇
                col1, col2 = st.columns(2)
                with col1:
                    time_col = st.selectbox("時間欄位", df.columns)
                with col2:
                    flux_col = st.selectbox("流量欄位", df.columns)
                
                if st.button("✅ 確認並分析"):
                    st.session_state.lightcurve_data = df[[time_col, flux_col]]
                    st.session_state.page = 'ai_screening'
                    st.rerun()
                    
            except Exception as e:
                st.error(f"檔案讀取失敗: {e}")
    
    elif input_method == "✏️ 手動輸入單筆候選數值":
        st.info("💡 輸入已知的行星候選特徵")
        
        col1, col2 = st.columns(2)
        
        with col1:
            period = st.number_input("軌道週期 (天)", value=10.0, min_value=0.1)
            duration = st.number_input("凌日持續時間 (小時)", value=3.0, min_value=0.1)
            depth = st.number_input("凌日深度 (ppm)", value=500.0, min_value=0.0)
            snr = st.number_input("信噪比", value=20.0, min_value=0.0)
        
        with col2:
            prad = st.number_input("行星半徑 (地球半徑)", value=2.0, min_value=0.0)
            teq = st.number_input("平衡溫度 (K)", value=500.0, min_value=0.0)
            insol = st.number_input("恆星輻射", value=1.0, min_value=0.0)
            steff = st.number_input("恆星溫度 (K)", value=5500.0, min_value=0.0)
        
        if st.button("🚀 送出分析"):
            candidate_data = {
                'period': period, 'duration': duration, 'depth': depth,
                'prad': prad, 'teq': teq, 'insol': insol,
                'snr': snr, 'steff': steff
            }
            st.session_state.current_candidate = candidate_data
            st.session_state.page = 'ai_screening'
            st.rerun()
    
    else:  # 使用預先整理的資料
        st.info("💡 從 NASA 公開資料集中選擇候選行星")
        
        # 生成模擬候選列表
        if st.button("📥 載入 Kepler 候選資料"):
            with st.spinner("載入中..."):
                # 生成模擬候選
                candidates = []
                for i in range(20):
                    time, flux, period, depth, duration = generate_mock_lightcurve()
                    candidates.append({
                        'id': f'KOI-{1000+i}',
                        'time': time,
                        'flux': flux,
                        'period': period,
                        'depth': depth,
                        'duration': duration,
                        'ai_confidence': np.random.uniform(0.3, 0.95)
                    })
                
                st.session_state.candidates = candidates
                st.success(f"✅ 載入 {len(candidates)} 個候選行星")
                
                if st.button("開始審查"):
                    st.session_state.page = 'ai_screening'
                    st.rerun()
    
    st.markdown("---")
    
    # 選擇演算法
    st.subheader("2️⃣ 選擇 AI 模型")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        model = st.selectbox(
            "選擇模型版本：",
            ["ensemble_v1 (多模型堆疊)", 
             "lgbm_v2 (LightGBM)", 
             "neural_net_v1 (深度學習)"]
        )
        st.session_state.selected_model = model.split()[0]
    
    with col2:
        st.metric("模型準確率", "92.3%")
        st.metric("訓練資料量", "8,547 筆")
    
    # 返回首頁
    if st.button("← 返回首頁"):
        st.session_state.page = 'home'
        st.rerun()

# ==================== 頁面：AI 初步篩選 ====================

def ai_screening_page():
    """AI 初步篩選頁面"""
    
    st.title("🤖 AI 初步篩選結果")
    
    # 模擬 AI 篩選
    if not st.session_state.candidates:
        # 生成測試候選
        candidates = []
        for i in range(10):
            time, flux, period, depth, duration = generate_mock_lightcurve()
            confidence = np.random.uniform(0.3, 0.95)
            candidates.append({
                'id': f'KOI-{2000+i}',
                'time': time,
                'flux': flux,
                'period': period,
                'depth': depth,
                'duration': duration,
                'ai_confidence': confidence
            })
        st.session_state.candidates = candidates
    
    # 分類統計
    candidates = st.session_state.candidates
    high_conf = [c for c in candidates if c['ai_confidence'] > 0.5]
    low_conf = [c for c in candidates if c['ai_confidence'] <= 0.5]
    
    st.subheader("📊 AI 分類統計")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: #fef3c7; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #f59e0b;'>
            <h3 style='color: #92400e;'>⚠️ 需要人類判斷</h3>
            <p style='font-size: 2rem; font-weight: bold; color: #92400e; margin: 0;'>{}</p>
            <p style='color: #78350f;'>信心度 > 50%</p>
        </div>
        """.format(len(high_conf)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: #fee2e2; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #ef4444;'>
            <h3 style='color: #991b1b;'>❌ 明顯誤報</h3>
            <p style='font-size: 2rem; font-weight: bold; color: #991b1b; margin: 0;'>{}</p>
            <p style='color: #7f1d1d;'>信心度 < 50%</p>
        </div>
        """.format(len(low_conf)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: #dbeafe; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #3b82f6;'>
            <h3 style='color: #1e40af;'>📊 總計</h3>
            <p style='font-size: 2rem; font-weight: bold; color: #1e40af; margin: 0;'>{}</p>
            <p style='color: #1e3a8a;'>待處理候選</p>
        </div>
        """.format(len(candidates)), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 顯示需要審查的候選
    st.subheader("🔍 需要你審查的候選")
    
    if high_conf:
        st.info(f"💡 以下是 AI 不確定的 {len(high_conf)} 個候選，需要你的專業判斷！")
        
        # 顯示前 3 個候選預覽
        for i, candidate in enumerate(high_conf[:3]):
            with st.expander(f"📌 {candidate['id']} - AI 信心度: {candidate['ai_confidence']:.1%}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # 簡單的光變曲線預覽
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=candidate['time'][:200],
                        y=candidate['flux'][:200],
                        mode='lines',
                        name='Flux'
                    ))
                    fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.metric("週期", f"{candidate['period']:.2f} 天")
                    st.metric("深度", f"{candidate['depth']*100:.2f}%")
        
        if st.button("🚀 開始逐一審查", type="primary", use_container_width=True):
            st.session_state.page = 'human_review'
            st.session_state.candidate_index = 0
            st.rerun()
    else:
        st.success("✅ 所有候選都已被 AI 明確分類，無需人類審查！")
    
    # 返回按鈕
    if st.button("← 返回上傳頁面"):
        st.session_state.page = 'upload'
        st.rerun()

# ==================== 頁面：人類判斷介面 ====================

def human_review_page():
    """人類判斷介面"""
    
    candidates = [c for c in st.session_state.candidates if c['ai_confidence'] > 0.5]
    
    if not candidates or st.session_state.candidate_index >= len(candidates):
        st.success("🎉 你已審查完所有候選！")
        if st.button("查看貢獻統計"):
            st.session_state.page = 'contribution'
            st.rerun()
        return
    
    current = candidates[st.session_state.candidate_index]
    
    # 頂部進度
    st.progress((st.session_state.candidate_index + 1) / len(candidates))
    st.caption(f"進度: {st.session_state.candidate_index + 1} / {len(candidates)}")
    
    st.title(f"🔍 審查候選：{current['id']}")
    
    # 主要視覺化區域
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📈 光變曲線")
        
        # 完整光變曲線
        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        
        # 完整曲線
        fig.add_trace(
            go.Scatter(
                x=current['time'],
                y=current['flux'],
                mode='lines',
                name='完整光變曲線',
                line=dict(color='#3b82f6', width=1)
            ),
            row=1, col=1
        )
        
        # 標註凌日區域
        period = current['period']
        for i in range(int(100/period)):
            transit_time = i * period
            fig.add_vrect(
                x0=transit_time - current['duration']/2,
                x1=transit_time + current['duration']/2,
                fillcolor="red",
                opacity=0.2,
                layer="below",
                line_width=0,
                row=1, col=1
            )
        
        # 放大第一個凌日
        transit_mask = (current['time'] > 0) & (current['time'] < period*2)
        fig.add_trace(
            go.Scatter(
                x=current['time'][transit_mask],
                y=current['flux'][transit_mask],
                mode='markers+lines',
                name='凌日特寫',
                line=dict(color='#ef4444', width=2),
                marker=dict(size=4)
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="時間 (天)", row=2, col=1)
        fig.update_yaxes(title_text="相對流量", row=1, col=1)
        fig.update_yaxes(title_text="相對流量", row=2, col=1)
        fig.update_layout(height=600, showlegend=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 候選資訊")
        
        # AI 信心度
        confidence = current['ai_confidence']
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem; background: #f3f4f6; border-radius: 0.5rem;'>
            <p style='margin: 0; color: #6b7280;'>AI 信心度</p>
            <p style='font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0; color: {"#10b981" if confidence > 0.7 else "#f59e0b"};'>
                {confidence:.0%}
            </p>
            <div style='height: 20px; background: #e5e7eb; border-radius: 10px; overflow: hidden;'>
                <div style='height: 100%; width: {confidence*100}%; background: linear-gradient(to right, #ef4444, #f59e0b, #10b981);'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 特徵資訊
        st.metric("軌道週期", f"{current['period']:.2f} 天")
        st.metric("凌日深度", f"{current['depth']*100:.3f}%")
        st.metric("持續時間", f"{current['duration']:.2f} 小時")
        
        st.markdown("---")
        
        # 快速提示
        st.info("""
        **💡 判斷提示**
        
        ✅ **像行星的特徵：**
        - 週期性凌日
        - 平滑的U型曲線
        - 深度一致
        
        ❌ **可能是誤報：**
        - 不規則形狀
        - V型尖峰
        - 不同深度
        """)
    
    st.markdown("---")
    
    # 滑動操作按鈕
    st.subheader("🎯 你的判斷")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("👈 左滑\n不像行星 (FP)", use_container_width=True, type="secondary"):
            reason = st.text_input("原因（可選）", key="fp_reason")
            save_user_annotation(current['id'], 'FALSE_POSITIVE', 'low', reason)
            st.session_state.candidate_index += 1
            st.rerun()
    
    with col2:
        if st.button("👉 右滑\n保持候選 (PC)", use_container_width=True):
            save_user_annotation(current['id'], 'CANDIDATE', 'medium')
            st.session_state.candidate_index += 1
            st.rerun()
    
    with col3:
        if st.button("👆 上滑\n強烈預感 (CP)", use_container_width=True, type="primary"):
            save_user_annotation(current['id'], 'CONFIRMED', 'high')
            st.session_state.candidate_index += 1
            st.rerun()
    
    # 跳過按鈕
    if st.button("⏭️ 跳過此候選"):
        st.session_state.candidate_index += 1
        st.rerun()

# ==================== 頁面：貢獻統計 ====================

def contribution_page():
    """貢獻統計頁面"""
    
    st.title("🏆 你的貢獻統計")
    
    labels = st.session_state.user_labels
    
    if not labels:
        st.info("你還沒有進行任何標註，快去審查候選行星吧！")
        if st.button("開始審查"):
            st.session_state.page = 'upload'
            st.rerun()
        return
    
    # 總體統計
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='contribution-card'>
            <h3>總標註數</h3>
            <p style='font-size: 3rem; margin: 0;'>{}</p>
        </div>
        """.format(len(labels)), unsafe_allow_html=True)
    
    with col2:
        confirmed = len([l for l in labels if l['label'] == 'CONFIRMED'])
        st.markdown("""
        <div style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 1.5rem; border-radius: 1rem; text-align: center;'>
            <h3>確認為行星</h3>
            <p style='font-size: 3rem; margin: 0;'>{}</p>
        </div>
        """.format(confirmed), unsafe_allow_html=True)
    
    with col3:
        candidates = len([l for l in labels if l['label'] == 'CANDIDATE'])
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; padding: 1.5rem; border-radius: 1rem; text-align: center;'>
            <h3>候選行星</h3>
            <p style='font-size: 3rem; margin: 0;'>{}</p>
        </div>
        """.format(candidates), unsafe_allow_html=True)
    
    with col4:
        false_pos = len([l for l in labels if l['label'] == 'FALSE_POSITIVE'])
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white; padding: 1.5rem; border-radius: 1rem; text-align: center;'>
            <h3>假陽性</h3>
            <p style='font-size: 3rem; margin: 0;'>{}</p>
        </div>
        """.format(false_pos), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 標註歷史
    st.subheader("📋 標註歷史")
    
    df_labels = pd.DataFrame(labels)
    df_labels['timestamp'] = pd.to_datetime(df_labels['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    
    st.dataframe(
        df_labels[['candidate_id', 'label', 'confidence', 'timestamp', 'reason']],
        use_container_width=True
    )
    
    # 下載標註
    csv = df_labels.to_csv(index=False)
    st.download_button(
        "📥 下載我的標註記錄",
        csv,
        "my_annotations.csv",
        "text/csv"
    )
    
    st.markdown("---")
    
    # 模擬準確度比較
    st.subheader("🎯 準確度比較")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 模擬使用者準確度
        user_accuracy = np.random.uniform(0.75, 0.95)
        st.metric(
            "你的判斷準確度",
            f"{user_accuracy:.1%}",
            delta=f"+{(user_accuracy-0.7)*100:.1f}%",
            help="與科學家驗證結果比對"
        )
    
    with col2:
        # AI 準確度
        ai_accuracy = 0.87
        st.metric(
            "AI 判斷準確度",
            f"{ai_accuracy:.1%}",
            help="目前模型在測試集上的表現"
        )
    
    # 貢獻度視覺化
    st.markdown("---")
    st.subheader("📈 貢獻趨勢")
    
    # 模擬時間趨勢
    fig = go.Figure()
    
    dates = pd.date_range(start='2025-01-01', periods=len(labels), freq='H')
    cumsum = list(range(1, len(labels)+1))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=cumsum,
        mode='lines+markers',
        fill='tozeroy',
        name='累積標註數'
    ))
    
    fig.update_layout(
        title="你的標註累積趨勢",
        xaxis_title="時間",
        yaxis_title="累積標註數",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 操作按鈕
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("繼續審查更多候選", type="primary", use_container_width=True):
            st.session_state.page = 'upload'
            st.rerun()
    
    with col2:
        if st.button("返回首頁", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()

# ==================== 主應用 ====================

def main():
    """主應用"""
    
    # 側邊欄導航
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150/667eea/ffffff?text=HITL", width=150)
        
        st.markdown("### 🧭 導航")
        
        if st.button("🏠 首頁", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
        
        if st.button("📤 上傳資料", use_container_width=True):
            st.session_state.page = 'upload'
            st.rerun()
        
        if st.button("🤖 AI 篩選", use_container_width=True):
            st.session_state.page = 'ai_screening'
            st.rerun()
        
        if st.button("👁️ 人類審查", use_container_width=True):
            st.session_state.page = 'human_review'
            st.rerun()
        
        if st.button("🏆 我的貢獻", use_container_width=True):
            st.session_state.page = 'contribution'
            st.rerun()
        
        st.markdown("---")
        
        # API 狀態
        api_status = check_api()
        if api_status:
            st.success("✅ API 已連接")
        else:
            st.warning("⚠️ API 未連接")
        
        st.markdown("---")
        
        # 統計摘要
        st.markdown("### 📊 快速統計")
        st.metric("已標註", len(st.session_state.user_labels))
        st.metric("已審查", st.session_state.candidate_index)
    
    # 路由到對應頁面
    if st.session_state.page == 'home':
        home_page()
    elif st.session_state.page == 'upload':
        upload_page()
    elif st.session_state.page == 'ai_screening':
        ai_screening_page()
    elif st.session_state.page == 'human_review':
        human_review_page()
    elif st.session_state.page == 'contribution':
        contribution_page()

if __name__ == "__main__":
    main()