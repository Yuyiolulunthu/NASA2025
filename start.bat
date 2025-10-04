@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
cls

echo ╔═══════════════════════════════════════════════════════╗
echo ║     系外行星檢測系統 - 啟動程序                        ║
echo ║     Exoplanet Detection System - Startup              ║
echo ╚═══════════════════════════════════════════════════════╝
echo.

REM 檢查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 錯誤: 找不到 Python
    pause
    exit /b 1
)
echo ✓ Python 已安裝
echo.

REM 檢查虛擬環境
if not exist "venv\" (
    echo 📦 建立虛擬環境...
    python -m venv venv
    echo ✓ 虛擬環境建立完成
) else (
    echo ✓ 虛擬環境已存在
)

REM 啟動虛擬環境
echo 🔧 啟動虛擬環境...
call venv\Scripts\activate.bat
echo.

REM 安裝依賴
if not exist "venv\.installed" (
    echo 📦 安裝依賴套件...
    pip install -q -r requirements.txt
    type nul > venv\.installed
    echo ✓ 依賴套件安裝完成
) else (
    echo ✓ 依賴套件已安裝
)
echo.

echo ─────────────────────────────────────────────────────
echo.

REM 建立目錄
if not exist "models\" mkdir models
if not exist "logs\" mkdir logs

REM 檢查模型
set MODEL_EXISTS=0
for %%F in (models\*.pkl) do set MODEL_EXISTS=1

if !MODEL_EXISTS!==0 (
    echo ⚠️  警告: 找不到訓練好的模型
    echo.
    echo 選項:
    echo   1 = 生成範例資料並訓練模型 ^(推薦^)
    echo   2 = 僅生成範例資料
    echo   3 = 跳過 ^(API 將無法運作^)
    echo.
    set /p choice="請輸入選項 (1 或 2 或 3): "
    
    if "!choice!"=="1" (
        echo.
        echo 🔧 生成範例資料...
        python generate_sample_data.py
        echo.
        echo 🚀 開始訓練模型...
        python train_exoplanet_model.py
        echo.
    ) else if "!choice!"=="2" (
        echo.
        echo 🔧 生成範例資料...
        python generate_sample_data.py
        echo.
    ) else (
        echo.
        echo ⚠️  跳過訓練
        echo.
    )
) else (
    echo ✓ 模型已存在
)

echo.
echo ─────────────────────────────────────────────────────
echo.
echo 🚀 啟動服務...
echo.

REM 啟動 API
echo ▶️  啟動 FastAPI 服務...
start "Exoplanet API" /MIN cmd /c "cd /d %CD% && venv\Scripts\python.exe backend\app.py"

REM 等待 API 啟動
echo ⏳ 等待 API 啟動...
timeout /t 6 /nobreak >nul
echo ✓ API 服務已啟動

echo.

REM 啟動前端
echo ▶️  啟動 Streamlit 前端...
start "Exoplanet Frontend" cmd /c "cd /d %CD% && venv\Scripts\streamlit.exe run frontend\app.py --server.port 8501"

timeout /t 2 /nobreak >nul

echo.
echo ╔═══════════════════════════════════════════════════════╗
echo ║              🎉 系統啟動成功！                         ║
echo ╚═══════════════════════════════════════════════════════╝
echo.
echo 📍 服務位置:
echo    API:      http://localhost:8000
echo    API Docs: http://localhost:8000/docs
echo    前端:     http://localhost:8501
echo.
echo 💡 提示:
echo    - 服務已在新視窗中啟動
echo    - 前端將自動在瀏覽器開啟
echo    - 關閉那些視窗即可停止服務
echo    - 或執行 stop.bat 停止所有服務
echo.
pause