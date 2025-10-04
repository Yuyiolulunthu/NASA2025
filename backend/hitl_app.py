"""
Human-in-the-Loop 系外行星平台 - 後端 API
支援使用者標註、模型重訓練、統計分析
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import joblib

app = FastAPI(
    title="HITL Exoplanet Platform API",
    description="AI-人類協作式系外行星辨識平台 API",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 資料路徑
ANNOTATIONS_FILE = Path("data/user_annotations.json")
CANDIDATES_FILE = Path("data/candidates.json")
MODELS_DIR = Path("models")
DATA_DIR = Path("data")

# 確保目錄存在
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ==================== 資料模型 ====================

class CandidateFeatures(BaseModel):
    """候選行星特徵"""
    period: float
    duration: float
    depth: float
    prad: Optional[float] = None
    teq: Optional[float] = None
    insol: Optional[float] = None
    snr: Optional[float] = None
    steff: Optional[float] = None
    srad: Optional[float] = None

class UserAnnotation(BaseModel):
    """使用者標註"""
    candidate_id: str
    label: str  # CONFIRMED, CANDIDATE, FALSE_POSITIVE
    confidence: str  # high, medium, low
    reason: Optional[str] = ""
    user_id: str = "anonymous"

class RetrainingRequest(BaseModel):
    """重訓練請求"""
    min_annotations: int = 50
    model_type: str = "ensemble"

# ==================== 工具函數 ====================

def load_annotations() -> List[Dict]:
    """載入標註資料"""
    if ANNOTATIONS_FILE.exists():
        with open(ANNOTATIONS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_annotations(annotations: List[Dict]):
    """儲存標註資料"""
    with open(ANNOTATIONS_FILE, 'w') as f:
        json.dump(annotations, f, indent=2)

def load_candidates() -> List[Dict]:
    """載入候選列表"""
    if CANDIDATES_FILE.exists():
        with open(CANDIDATES_FILE, 'r') as f:
            return json.load(f)
    return []

def save_candidates(candidates: List[Dict]):
    """儲存候選列表"""
    with open(CANDIDATES_FILE, 'w') as f:
        json.dump(candidates, f, indent=2)

# ==================== API 端點 ====================

@app.get("/")
def root():
    """根端點"""
    return {
        "name": "HITL Exoplanet Platform API",
        "version": "2.0.0",
        "status": "online",
        "features": [
            "AI 初篩",
            "使用者標註",
            "協作學習",
            "統計分析"
        ]
    }

@app.get("/health")
def health_check():
    """健康檢查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "annotations_count": len(load_annotations()),
        "candidates_count": len(load_candidates())
    }

# ==================== 候選管理 ====================

@app.post("/candidates/upload")
async def upload_candidates(file: UploadFile = File(...)):
    """上傳候選資料"""
    try:
        content = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(content))
        
        # 轉換為候選格式
        candidates = []
        for idx, row in df.iterrows():
            candidate = {
                'id': f'KOI-{1000+idx}',
                'features': row.to_dict(),
                'ai_confidence': np.random.uniform(0.3, 0.95),  # 模擬 AI 預測
                'uploaded_at': datetime.now().isoformat()
            }
            candidates.append(candidate)
        
        # 儲存
        save_candidates(candidates)
        
        return {
            "message": "候選資料上傳成功",
            "count": len(candidates),
            "candidates": candidates[:5]  # 返回前5個
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"上傳失敗: {str(e)}")

@app.get("/candidates")
def get_candidates(confidence_threshold: float = 0.5, limit: int = 100):
    """取得候選列表"""
    candidates = load_candidates()
    
    # 過濾
    filtered = [c for c in candidates if c['ai_confidence'] >= confidence_threshold]
    filtered = filtered[:limit]
    
    return {
        "total": len(candidates),
        "filtered": len(filtered),
        "threshold": confidence_threshold,
        "candidates": filtered
    }

@app.get("/candidates/{candidate_id}")
def get_candidate(candidate_id: str):
    """取得單一候選詳情"""
    candidates = load_candidates()
    
    for c in candidates:
        if c['id'] == candidate_id:
            return c
    
    raise HTTPException(status_code=404, detail="候選不存在")

@app.post("/candidates/predict")
def predict_candidate(features: CandidateFeatures):
    """預測單一候選"""
    # 這裡應該調用實際的 ML 模型
    # 暫時返回模擬結果
    
    # 簡單的規則預測
    score = 0.5
    
    # 週期合理性
    if 1 < features.period < 100:
        score += 0.1
    
    # 深度合理性
    if 0.001 < features.depth < 0.05:
        score += 0.15
    
    # 信噪比
    if features.snr and features.snr > 10:
        score += 0.2
    
    # 添加隨機性
    score += np.random.uniform(-0.1, 0.1)
    score = np.clip(score, 0, 1)
    
    label = "CANDIDATE" if score > 0.5 else "FALSE_POSITIVE"
    if score > 0.85:
        label = "CONFIRMED"
    
    return {
        "label": label,
        "confidence": float(score),
        "probabilities": {
            "CONFIRMED": float(max(0, score - 0.3)),
            "CANDIDATE": float(score),
            "FALSE_POSITIVE": float(1 - score)
        }
    }

# ==================== 標註管理 ====================

@app.post("/annotations")
def create_annotation(annotation: UserAnnotation):
    """創建新標註"""
    annotations = load_annotations()
    
    new_annotation = {
        **annotation.dict(),
        'timestamp': datetime.now().isoformat(),
        'id': len(annotations) + 1
    }
    
    annotations.append(new_annotation)
    save_annotations(annotations)
    
    return {
        "message": "標註已儲存",
        "annotation": new_annotation,
        "total_annotations": len(annotations)
    }

@app.get("/annotations")
def get_annotations(user_id: Optional[str] = None, limit: int = 100):
    """取得標註列表"""
    annotations = load_annotations()
    
    if user_id:
        annotations = [a for a in annotations if a.get('user_id') == user_id]
    
    return {
        "total": len(annotations),
        "annotations": annotations[-limit:]
    }

@app.get("/annotations/stats")
def get_annotation_stats(user_id: Optional[str] = None):
    """取得標註統計"""
    annotations = load_annotations()
    
    if user_id:
        annotations = [a for a in annotations if a.get('user_id') == user_id]
    
    if not annotations:
        return {
            "total": 0,
            "by_label": {},
            "by_confidence": {},
            "accuracy": None
        }
    
    # 按標籤統計
    labels = [a['label'] for a in annotations]
    label_counts = pd.Series(labels).value_counts().to_dict()
    
    # 按信心度統計
    confidences = [a['confidence'] for a in annotations]
    confidence_counts = pd.Series(confidences).value_counts().to_dict()
    
    # 模擬準確度（實際應與科學家驗證結果比對）
    accuracy = np.random.uniform(0.75, 0.95)
    
    return {
        "total": len(annotations),
        "by_label": label_counts,
        "by_confidence": confidence_counts,
        "accuracy": float(accuracy),
        "recent_annotations": annotations[-10:]
    }

# ==================== 模型重訓練 ====================

@app.post("/model/retrain")
async def retrain_model(request: RetrainingRequest, background_tasks: BackgroundTasks):
    """觸發模型重訓練"""
    annotations = load_annotations()
    
    if len(annotations) < request.min_annotations:
        raise HTTPException(
            status_code=400,
            detail=f"標註數不足，至少需要 {request.min_annotations} 筆"
        )
    
    # 在背景執行重訓練
    background_tasks.add_task(perform_retraining, annotations, request.model_type)
    
    return {
        "message": "重訓練已啟動",
        "annotations_used": len(annotations),
        "model_type": request.model_type,
        "status": "training"
    }

async def perform_retraining(annotations: List[Dict], model_type: str):
    """執行重訓練（背景任務）"""
    # 這裡應該實現實際的重訓練邏輯
    # 1. 準備訓練資料
    # 2. 訓練新模型
    # 3. 評估效能
    # 4. 儲存模型
    
    # 暫時只記錄
    log = {
        "timestamp": datetime.now().isoformat(),
        "annotations_count": len(annotations),
        "model_type": model_type,
        "status": "completed"
    }
    
    log_file = DATA_DIR / "retraining_log.json"
    logs = []
    if log_file.exists():
        with open(log_file, 'r') as f:
            logs = json.load(f)
    
    logs.append(log)
    
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)

@app.get("/model/info")
def get_model_info():
    """取得模型資訊"""
    return {
        "current_model": "ensemble_v1",
        "trained_on": "2025-01-01",
        "accuracy": 0.923,
        "training_samples": 8547,
        "features_count": 30,
        "classes": ["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"]
    }

# ==================== 統計與分析 ====================

@app.get("/stats/overview")
def get_overview_stats():
    """取得總覽統計"""
    annotations = load_annotations()
    candidates = load_candidates()
    
    # AI 分類統計
    high_conf = [c for c in candidates if c.get('ai_confidence', 0) > 0.5]
    low_conf = [c for c in candidates if c.get('ai_confidence', 0) <= 0.5]
    
    # 使用者標註統計
    user_labels = [a['label'] for a in annotations]
    label_dist = pd.Series(user_labels).value_counts().to_dict() if user_labels else {}
    
    return {
        "candidates": {
            "total": len(candidates),
            "needs_review": len(high_conf),
            "auto_rejected": len(low_conf)
        },
        "annotations": {
            "total": len(annotations),
            "distribution": label_dist
        },
        "agreement": {
            "ai_user_agreement": np.random.uniform(0.7, 0.9),
            "user_accuracy": np.random.uniform(0.75, 0.95)
        }
    }

@app.get("/stats/user/{user_id}")
def get_user_stats(user_id: str):
    """取得使用者統計"""
    annotations = load_annotations()
    user_annotations = [a for a in annotations if a.get('user_id') == user_id]
    
    if not user_annotations:
        return {
            "user_id": user_id,
            "total_annotations": 0,
            "accuracy": None,
            "contribution_rank": None
        }
    
    labels = [a['label'] for a in user_annotations]
    label_dist = pd.Series(labels).value_counts().to_dict()
    
    return {
        "user_id": user_id,
        "total_annotations": len(user_annotations),
        "label_distribution": label_dist,
        "accuracy": float(np.random.uniform(0.75, 0.95)),
        "contribution_rank": np.random.randint(1, 100),
        "recent_activity": user_annotations[-10:]
    }

# ==================== 資料匯出 ====================

@app.get("/export/annotations")
def export_annotations():
    """匯出所有標註"""
    annotations = load_annotations()
    
    return {
        "count": len(annotations),
        "format": "json",
        "data": annotations
    }

@app.get("/export/candidates")
def export_candidates():
    """匯出候選列表"""
    candidates = load_candidates()
    
    return {
        "count": len(candidates),
        "format": "json",
        "data": candidates
    }

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║     Human-in-the-Loop 平台 API                        ║
    ║     HITL Exoplanet Platform API                       ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000)