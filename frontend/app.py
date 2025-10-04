"""
系外行星檢測 - 簡化版前端
確保基本功能可用
"""

import streamlit as st

try:
    import requests
    HAS_REQUESTS = True
except:
    HAS_REQUESTS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except:
    HAS_PANDAS = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except:
    HAS_PLOTLY = False

# 頁面配置
st.set_page_config(
    page_title="Exoplanet Detection",
    page_icon="🪐",
    layout="wide"
)

# 標題
st.title("🪐 系外行星檢測系統")
st.markdown("---")

# 檢查依賴
with st.expander("📦 系統狀態檢查"):
    st.write("依賴套件狀態：")
    st.write(f"- requests: {'✅ 已安裝' if HAS_REQUESTS else '❌ 未安裝'}")
    st.write(f"- pandas: {'✅ 已安裝' if HAS_PANDAS else '❌ 未安裝'}")
    st.write(f"- plotly: {'✅ 已安裝' if HAS_PLOTLY else '❌ 未安裝'}")
    
    if not all([HAS_REQUESTS, HAS_PANDAS, HAS_PLOTLY]):
        st.error("請安裝缺少的套件：pip install requests pandas plotly")

# API 配置
API_URL = "http://localhost:8000"

# 側邊欄
st.sidebar.title("🎛️ 功能選單")
page = st.sidebar.radio(
    "選擇功能",
    ["🏠 首頁", "🔍 單筆預測", "📊 批次預測", "📖 說明"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
### 系統簡介
使用 AI 識別系外行星：
- **Confirmed**: 已確認
- **Candidate**: 候選
- **False Positive**: 假陽性
""")

# 檢查 API 連線
def check_api():
    """檢查 API 是否可用"""
    if not HAS_REQUESTS:
        return False, "requests 套件未安裝"
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            return True, "API 正常"
        else:
            return False, f"API 錯誤: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "無法連接 API (請確認 backend/app.py 是否運行)"
    except Exception as e:
        return False, f"錯誤: {str(e)}"

# 主要內容
if page == "🏠 首頁":
    st.header("歡迎使用系外行星檢測系統")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 系統功能")
        st.write("""
        - ✨ AI 驅動的行星識別
        - 📈 高準確度分類
        - 🔍 單筆快速預測
        - 📊 批次資料處理
        - 📉 視覺化分析
        """)
    
    with col2:
        st.subheader("🔧 系統狀態")
        api_ok, api_msg = check_api()
        
        if api_ok:
            st.success(f"✅ {api_msg}")
            st.info("系統就緒，可以開始使用！")
        else:
            st.error(f"❌ {api_msg}")
            st.warning("""
            請確認：
            1. 已執行 `python backend/app.py`
            2. API 運行在 http://localhost:8000
            """)
    
    st.markdown("---")
    st.subheader("📚 快速開始")
    st.write("""
    1. 確認左側狀態顯示「API 正常」
    2. 點選左側選單的「🔍 單筆預測」
    3. 輸入行星特徵
    4. 查看預測結果
    """)

elif page == "🔍 單筆預測":
    st.header("🔍 單筆預測")
    
    # 檢查 API
    api_ok, api_msg = check_api()
    if not api_ok:
        st.error(f"⚠️ {api_msg}")
        st.stop()
    
    st.success("✅ API 連線正常")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("輸入行星特徵")
        
        with st.form("prediction_form"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                period = st.number_input("軌道週期 (天)", value=10.0, min_value=0.1)
                duration = st.number_input("凌日持續時間 (小時)", value=3.0, min_value=0.1)
                depth = st.number_input("凌日深度 (ppm)", value=500.0, min_value=0.0)
                snr = st.number_input("信噪比", value=20.0, min_value=0.0)
            
            with col_b:
                prad = st.number_input("行星半徑 (地球)", value=2.0, min_value=0.0)
                teq = st.number_input("平衡溫度 (K)", value=500.0, min_value=0.0)
                insol = st.number_input("恆星輻射", value=1.0, min_value=0.0)
                steff = st.number_input("恆星溫度 (K)", value=5500.0, min_value=0.0)
            
            srad = st.number_input("恆星半徑 (太陽)", value=1.0, min_value=0.0)
            
            submitted = st.form_submit_button("🚀 開始預測", use_container_width=True)
        
        if submitted:
            with st.spinner("預測中..."):
                data = {
                    "koi_period": period,
                    "koi_duration": duration,
                    "koi_depth": depth,
                    "koi_prad": prad,
                    "koi_teq": teq,
                    "koi_insol": insol,
                    "koi_model_snr": snr,
                    "koi_steff": steff,
                    "koi_srad": srad
                }
                
                try:
                    response = requests.post(f"{API_URL}/predict", json=data)
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state['result'] = result
                        st.success("✅ 預測完成！")
                    else:
                        st.error(f"預測失敗: {response.text}")
                except Exception as e:
                    st.error(f"錯誤: {e}")
    
    with col2:
        st.subheader("預測結果")
        
        if 'result' in st.session_state:
            result = st.session_state['result']
            label = result['label']
            confidence = result['confidence']
            
            # 顯示結果
            if label == 'CONFIRMED':
                st.success(f"### 🪐 {label}")
                color = "green"
            elif label == 'CANDIDATE':
                st.warning(f"### 🔍 {label}")
                color = "orange"
            else:
                st.info(f"### ❌ {label}")
                color = "blue"
            
            st.metric("信心度", f"{confidence:.2%}")
            
            # 機率分佈
            if HAS_PANDAS and HAS_PLOTLY:
                st.subheader("機率分佈")
                probs = result['probabilities']
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(probs.keys()),
                        y=list(probs.values()),
                        text=[f"{v:.1%}" for v in probs.values()],
                        textposition='auto',
                    )
                ])
                fig.update_layout(
                    yaxis_title="機率",
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 請在左側輸入特徵並預測")

elif page == "📊 批次預測":
    st.header("📊 批次預測")
    
    # 檢查 API
    api_ok, api_msg = check_api()
    if not api_ok:
        st.error(f"⚠️ {api_msg}")
        st.stop()
    
    st.info("""
    上傳包含以下欄位的 CSV 檔案：
    - koi_period, koi_duration, koi_depth
    - koi_prad, koi_teq, koi_insol
    - koi_model_snr, koi_steff, koi_srad
    """)
    
    uploaded_file = st.file_uploader("選擇 CSV 檔案", type=['csv'])
    
    if uploaded_file:
        if HAS_PANDAS:
            df = pd.read_csv(uploaded_file)
            st.write("**資料預覽：**")
            st.dataframe(df.head())
            
            if st.button("🚀 開始批次預測"):
                with st.spinner("處理中..."):
                    try:
                        uploaded_file.seek(0)
                        files = {"file": uploaded_file}
                        response = requests.post(f"{API_URL}/predict-batch", files=files)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"✅ 完成！共處理 {result['total']} 筆")
                            
                            # 顯示摘要
                            summary = result['summary']
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Confirmed", summary.get('CONFIRMED', 0))
                            col2.metric("Candidate", summary.get('CANDIDATE', 0))
                            col3.metric("False Positive", summary.get('FALSE POSITIVE', 0))
                            
                            # 顯示結果
                            st.write("**詳細結果：**")
                            predictions = result['predictions']
                            result_df = pd.DataFrame([
                                {
                                    'Index': p['index'],
                                    'Label': p.get('label', 'Error'),
                                    'Confidence': f"{p.get('confidence', 0):.2%}"
                                }
                                for p in predictions
                            ])
                            st.dataframe(result_df)
                        else:
                            st.error(f"錯誤: {response.text}")
                    except Exception as e:
                        st.error(f"錯誤: {e}")
        else:
            st.error("請安裝 pandas: pip install pandas")

else:  # 說明頁面
    st.header("📖 使用說明")
    
    st.markdown("""
    ## 系統概述
    
    這是一個基於機器學習的系外行星檢測系統，使用 NASA 的 Kepler、K2 和 TESS 任務資料訓練。
    
    ### 分類說明
    
    - **CONFIRMED** 🪐: 已確認的系外行星
    - **CANDIDATE** 🔍: 候選系外行星（需進一步驗證）
    - **FALSE POSITIVE** ❌: 假陽性（非行星信號）
    
    ### 特徵說明
    
    | 特徵 | 說明 |
    |------|------|
    | 軌道週期 | 行星繞恆星一圈的時間（天）|
    | 凌日持續時間 | 行星通過恆星表面的時間（小時）|
    | 凌日深度 | 亮度下降程度（ppm）|
    | 信噪比 | 信號品質指標 |
    | 行星半徑 | 相對地球半徑 |
    | 平衡溫度 | 行星溫度（K）|
    | 恆星輻射 | 接收的輻射量 |
    | 恆星溫度 | 恆星有效溫度（K）|
    | 恆星半徑 | 相對太陽半徑 |
    
    ### 技術架構
    
    - **後端**: FastAPI
    - **前端**: Streamlit
    - **模型**: 多模型堆疊集成
    - **特徵**: BLS/TLS + 統計特徵
    
    ### 資料來源
    
    [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
    """)

# 頁腳
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    🪐 Exoplanet Detection System | Built with Streamlit & FastAPI
</div>
""", unsafe_allow_html=True)