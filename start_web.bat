@echo off
cls

echo ====================================================
echo      InterTzailer Launcher
echo ====================================================
echo.

if exist "venv_nasa\" (
    call venv_nasa\Scripts\activate.bat
    echo [OK] Virtual environment activated
) else (
    echo [ERROR] venv_nasa folder not found
    pause
    exit /b 1
)

echo.
echo Creating necessary folders...
if not exist "data\" mkdir data
if not exist "models\" mkdir models
if not exist "logs\" mkdir logs

echo.
echo Starting Streamlit Dashboard...
echo.

start "Streamlit Dashboard" cmd /k "python -m streamlit run Web\main.py --server.port 8501"

timeout /t 5 /nobreak >nul

echo.
echo ====================================================
echo     Dashboard starting...
echo ====================================================
echo.
echo Dashboard: http://localhost:8501
echo.

timeout /t 3 /nobreak >nul
start http://localhost:8501

pause