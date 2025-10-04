@echo off
chcp 65001 >nul
cls

echo ╔═══════════════════════════════════════════════════════╗
echo ║     InterTzailer - 完整啟動                           ║
echo ║     Landing Page + Streamlit Dashboard               ║
echo ╚═══════════════════════════════════════════════════════╝
echo.

REM 啟動虛擬環境
if exist "venv\" (
    call venv\Scripts\activate.bat
) else (
    echo ⚠️  虛擬環境不存在
    pause
    exit
)

echo ✓ 虛擬環境已啟動
echo.

REM 建立必要目錄
if not exist "data\" mkdir data
if not exist "models\" mkdir models
if not exist "logs\" mkdir logs

echo ─────────────────────────────────────────────────────
echo   🚀 啟動服務
echo ─────────────────────────────────────────────────────
echo.

REM 啟動簡單的 HTTP 服務器 (著陸頁)
echo ▶️  啟動著陸頁 (port 8080)...
start "Landing Page" /MIN python -m http.server 8080

REM 等待一下
timeout /t 2 /nobreak >nul

REM 啟動 Streamlit
echo ▶️  啟動 Streamlit Dashboard (port 8501)...
start "Streamlit Dashboard" cmd /c "venv\Scripts\streamlit.exe run Web\main.py --server.port 8501"

timeout /t 3 /nobreak >nul

echo.
echo ╔═══════════════════════════════════════════════════════╗
echo ║              🎉 系統啟動成功！                         ║
echo ╚═══════════════════════════════════════════════════════╝
echo.
echo 📍 服務位置:
echo    著陸頁:    http://localhost:8080
echo    Dashboard: http://localhost:8501
echo.
echo 💡 使用說明:
echo    1. 訪問 http://localhost:8080 查看著陸頁
echo    2. 點擊 "Launch Dashboard" 進入 Streamlit 應用
echo    3. 開始審查系外行星候選！
echo.
echo ⏹️  停止: 關閉彈出的視窗
echo.

REM 自動開啟瀏覽器
timeout /t 2 /nobreak >nul
start http://localhost:8080

pause