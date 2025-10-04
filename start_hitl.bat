@echo off
chcp 65001 >nul
cls

echo ╔═══════════════════════════════════════════════════════╗
echo ║   AI-人類協作式系外行星辨識平台 - 啟動程序            ║
echo ║   HITL Exoplanet Platform - Startup                  ║
echo ╚═══════════════════════════════════════════════════════╝
echo.

REM 檢查虛擬環境
if not exist "venv\" (
    echo ⚠️  虛擬環境不存在，正在建立...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo 📦 安裝依賴套件...
    pip install streamlit fastapi uvicorn plotly pandas numpy requests pydantic
) else (
    call venv\Scripts\activate.bat
)

echo ✓ 環境已就緒
echo.

REM 建立必要目錄
if not exist "data\" mkdir data
if not exist "models\" mkdir models
if not exist "logs\" mkdir logs

echo ─────────────────────────────────────────────────────
echo   🚀 啟動服務
echo ─────────────────────────────────────────────────────
echo.

REM 啟動 API
echo ▶️  啟動 HITL API 服務...
start "HITL API" /MIN cmd /c "cd /d %CD% && venv\Scripts\python.exe backend\hitl_api.py"

REM 等待 API 啟動
echo ⏳ 等待 API 啟動...
timeout /t 5 /nobreak >nul
echo ✓ API 服務運行中: http://localhost:8000

echo.

REM 啟動前端
echo ▶️  啟動 HITL 前端界面...
start "HITL Frontend" cmd /c "cd /d %CD% && venv\Scripts\streamlit.exe run frontend\hitl_app.py --server.port 8501"

timeout /t 3 /nobreak >nul

echo.
echo ╔═══════════════════════════════════════════════════════╗
echo ║              🎉 平台啟動成功！                         ║
echo ╚═══════════════════════════════════════════════════════╝
echo.
echo 📍 服務位置:
echo    API:      http://localhost:8000
echo    API Docs: http://localhost:8000/docs
echo    前端:     http://localhost:8501
echo.
echo 🌌 Human-in-the-Loop 系外行星探索平台
echo.
echo 💡 使用說明:
echo    1. 瀏覽器將自動開啟前端界面
echo    2. 點擊「開始探索」開始使用
echo    3. 上傳資料或使用預設 NASA 資料
echo    4. 用滑動手勢審查候選行星
echo    5. 追蹤你的貢獻統計
echo.
echo 📖 完整說明: 查看 HITL_README.md
echo.
echo ⏹️  停止服務: 關閉彈出的視窗或執行 stop.bat
echo.
pause