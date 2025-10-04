"""簡化版 HITL 平台"""
import streamlit as st
import numpy as np

st.set_page_config(
    page_title="系外行星探索平台",
    page_icon="🌌",
    layout="wide"
)

# 初始化
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# 側邊欄
with st.sidebar:
    st.title("🧭 導航")
    if st.button("🏠 首頁"):
        st.session_state.page = 'home'
    if st.button("📤 上傳"):
        st.session_state.page = 'upload'
    if st.button("🔍 審查"):
        st.session_state.page = 'review'

# 首頁
if st.session_state.page == 'home':
    st.title("🌌 AI-人類協作式系外行星探索平台")
    
    st.markdown("""
    ### 🎯 我們的使命
    結合 AI 的速度與人類直覺的智慧，共同探索宇宙中的新世界！
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🤖 AI 初篩")
    with col2:
        st.success("👁️ 人類判斷")
    with col3:
        st.warning("🔄 協作學習")

# 上傳頁
elif st.session_state.page == 'upload':
    st.title("📤 資料上傳")
    
    method = st.radio("選擇方式", ["手動輸入", "上傳檔案"])
    
    if method == "手動輸入":
        period = st.number_input("軌道週期", value=10.0)
        depth = st.number_input("凌日深度", value=500.0)
        
        if st.button("分析"):
            st.success("✅ 已接收")

# 審查頁
else:
    st.title("🔍 候選審查")
    
    st.metric("AI 信心度", "67%")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("👈 不像"):
            st.success("已標註")
    with col2:
        if st.button("👉 候選"):
            st.info("已保持")
    with col3:
        if st.button("👆 確認"):
            st.balloons()