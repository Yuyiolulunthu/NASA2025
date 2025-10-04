#!/bin/bash

# 系外行星檢測系統 - 快速啟動腳本
# Exoplanet Detection System - Quick Start Script

echo "╔═══════════════════════════════════════════════════════╗"
echo "║     系外行星檢測系統 - 啟動程序                        ║"
echo "║     Exoplanet Detection System - Startup              ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# 檢查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 錯誤: 找不到 Python 3"
    echo "請先安裝 Python 3.8 或更高版本"
    exit 1
fi

echo "✓ Python 版本: $(python3 --version)"
echo ""

# 檢查虛擬環境
if [ ! -d "venv" ]; then
    echo "📦 建立虛擬環境..."
    python3 -m venv venv
    echo "✓ 虛擬環境建立完成"
else
    echo "✓ 虛擬環境已存在"
fi

# 啟動虛擬環境
echo "🔧 啟動虛擬環境..."
source venv/bin/activate

# 檢查並安裝依賴
echo "📦 檢查依賴套件..."
if [ ! -f "venv/.installed" ]; then
    echo "安裝依賴套件中（這可能需要幾分鐘）..."
    pip install -r requirements.txt
    touch venv/.installed
    echo "✓ 依賴套件安裝完成"
else
    echo "✓ 依賴套件已安裝"
fi

echo ""
echo "─────────────────────────────────────────────────────"
echo ""

# 檢查模型
if [ ! -d "models" ] || [ -z "$(ls -A models/*.pkl 2>/dev/null)" ]; then
    echo "⚠️  警告: 找不到訓練好的模型"
    echo ""
    read -p "是否要訓練模型？(y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🚀 開始訓練模型..."
        python train_exoplanet_model.py
        echo ""
    else
        echo "⚠️  跳過訓練，API 可能無法正常運作"
        echo ""
    fi
else
    echo "✓ 模型已存在"
fi

echo ""
echo "─────────────────────────────────────────────────────"
echo ""
echo "🚀 啟動服務..."
echo ""

# 建立日誌目錄
mkdir -p logs

# 啟動 API (背景執行)
echo "▶️  啟動 FastAPI 服務..."
python backend/app.py > logs/api.log 2>&1 &
API_PID=$!
echo "✓ FastAPI PID: $API_PID"

# 等待 API 啟動
echo "⏳ 等待 API 啟動..."
sleep 5

# 檢查 API 是否運行
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✓ API 服務運行中: http://localhost:8000"
    echo "✓ API 文件: http://localhost:8000/docs"
else
    echo "❌ API 啟動失敗，請檢查 logs/api.log"
    kill $API_PID 2>/dev/null
    exit 1
fi

echo ""
echo "▶️  啟動 Streamlit 前端..."
streamlit run frontend/app.py --server.port 8501 --server.headless true > logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "✓ Streamlit PID: $FRONTEND_PID"

echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║              🎉 系統啟動成功！                         ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""
echo "📍 服務位置:"
echo "   API:      http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo "   前端:     http://localhost:8501"
echo ""
echo "📋 服務管理:"
echo "   API PID:       $API_PID"
echo "   Frontend PID:  $FRONTEND_PID"
echo ""
echo "📊 日誌位置:"
echo "   API:      logs/api.log"
echo "   Frontend: logs/frontend.log"
echo ""
echo "⏹️  停止服務: kill $API_PID $FRONTEND_PID"
echo "   或執行: ./stop.sh"
echo ""
echo "前端將自動在瀏覽器開啟..."
echo "如果沒有自動開啟，請手動訪問: http://localhost:8501"
echo ""

# 儲存 PID
echo $API_PID > .api.pid
echo $FRONTEND_PID > .frontend.pid

# 等待使用者中斷
echo "按 Ctrl+C 停止服務..."
echo ""

# Trap Ctrl+C
trap ctrl_c INT

function ctrl_c() {
    echo ""
    echo "⏹️  停止服務中..."
    kill $API_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    rm -f .api.pid .frontend.pid
    echo "✓ 服務已停止"
    exit 0
}

# 保持腳本運行
wait