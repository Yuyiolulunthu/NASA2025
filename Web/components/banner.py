import streamlit as st

def render_banner():
    st.markdown("""
    <style>
    /* --- Fixed Top Banner --- */
    .top-banner {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 64px;
        background: rgba(10, 14, 39, 0.9);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(102, 126, 234, 0.3);
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 2rem;
        z-index: 1000;
    }

    /* --- Logo Section --- */
    .banner-left {
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }

    .banner-left img {
        width: 32px;
        height: 32px;
    }

    .banner-left span {
        color: #ffffff;
        font-size: 1.2rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }

    /* --- Hamburger Icon --- */
    .menu-icon {
        width: 25px;
        height: 2px;
        background-color: white;
        position: relative;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .menu-icon::before, .menu-icon::after {
        content: "";
        position: absolute;
        left: 0;
        width: 25px;
        height: 2px;
        background-color: white;
        transition: all 0.3s ease;
    }
    .menu-icon::before { top: -8px; }
    .menu-icon::after { top: 8px; }

    /* --- Slide-Out Menu --- */
    .menu-panel {
        position: fixed;
        top: 64px;
        right: -300px;
        width: 260px;
        background: rgba(22, 33, 62, 0.98);
        height: 100vh;
        transition: right 0.4s ease;
        padding: 1.5rem;
        border-left: 1px solid rgba(102, 126, 234, 0.3);
        z-index: 999;
    }
    .menu-panel.show {
        right: 0;
    }
    .menu-item {
        color: #e0e0e0;
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 1rem;
        cursor: pointer;
        transition: color 0.2s ease;
    }
    .menu-item:hover {
        color: #667eea;
    }
    .menu-sub {
        margin-left: 1rem;
    }
    </style>

    <!-- Banner Layout -->
    <div class="top-banner">
        <div class="banner-left">
            <img src="Web/logo.png" alt="logo">
            <span>ExoMatch</span>
        </div>
        <div class="banner-right">
            <div class="menu-icon" id="menu-btn"></div>
        </div>
    </div>

    <!-- Slide-out Menu -->
    <div class="menu-panel" id="menu-panel">
        <div class="menu-item">About our models</div>
        <div class="menu-item">Analyze your data</div>
        <div class="menu-sub">
            <div class="menu-item">Demo</div>
            <div class="menu-item">Analyze</div>
        </div>
        <div class="menu-item">Vet your data</div>
        <div class="menu-sub">
            <div class="menu-item">Demo</div>
            <div class="menu-item">Vetting</div>
        </div>
    </div>

    <script>
    const menuBtn = window.parent.document.getElementById("menu-btn");
    const panel = window.parent.document.getElementById("menu-panel");
    if (menuBtn && panel) {
        menuBtn.addEventListener("click", () => {
            panel.classList.toggle("show");
        });
    }
    </script>
    """, unsafe_allow_html=True)
