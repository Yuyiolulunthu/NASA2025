"""
系外行星檢測 API - FastAPI Backend
提供預測、解釋、批次處理等功能
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import joblib
import json
import shap
from pathlib import Path
from datetime import datetime
import io

app = FastAPI(
    title="Exoplanet Detection API",
    description="NASA 系外行星檢測 AI 模型 API",
    version="1.0.0"
)

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全域變數
MODEL = None
SCALER = None
LABEL_ENCODER = None
FEATURES = None
EXPLAINER = None


class FeatureInput(BaseModel):
    """單筆特徵輸入"""
    koi_period: float = Field(..., description="軌道週期 (天)")
    koi_duration: float = Field(..., description="凌日持續時間 (小時)")
    koi_depth: float = Field(..., description="凌日深度 (ppm)")
    koi_prad: Optional[float] = Field(None, description="行星半徑 (地球半徑)")
    koi_teq: Optional[float] = Field(None, description="平衡溫度 (K)")
    koi_insol: Optional[float] = Field(None, description="恆星輻射 (地球)")
    koi_model_snr: Optional[float] = Field(None, description="信噪比")
    koi_steff: Optional[float] = Field(None, description="恆星有效溫度 (K)")
    koi_srad: Optional[float] = Field(None, description="恆星半徑 (太陽半徑)")
    
    @validator('koi_period')
    def validate_period(cls, v):
        if v <= 0:
            raise ValueError('軌道週期必須大於 0')
        return v
    
    @validator('koi_depth')
    def validate_depth(cls, v):
        if v < 0:
            raise ValueError('凌日深度不能為負數')
        return v


class PredictionResponse(BaseModel):
    """預測結果"""
    label: str
    confidence: float
    probabilities: Dict[str, float]
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """批次預測結果"""
    total: int
    predictions: List[Dict]
    summary: Dict[str, int]
    timestamp: str


class ExplainResponse(BaseModel):
    """解釋結果"""
    prediction: str
    top_features: List[Dict[str, float]]
    shap_values: List[float]


@app.on_event("startup")
async def load_model():
    """啟動時載入模型"""
    global MODEL, SCALER, LABEL_ENCODER, FEATURES, EXPLAINER
    
    models_dir = Path("models")
    
    try:
        # 尋找最新的模型
        model_files = list(models_dir.glob("exoplanet_model_*.pkl"))
        if not model_files:
            print("⚠️  警告: 找不到訓練好的模型")
            print("請先執行: python train_exoplanet_model.py")
            return
        
        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
        
        print(f"📦 載入模型: {latest_model}")
        model_dict = joblib.load(latest_model)
        
        MODEL = model_dict['model']
        SCALER = model_dict['scaler']
        LABEL_ENCODER = model_dict['label_encoder']
        FEATURES = model_dict['selected_features']
        
        print(f"✅ 模型載入成功")
        print(f"   - 版本: {model_dict.get('version', 'unknown')}")
        print(f"   - 特徵數: {len(FEATURES)}")
        print(f"   - 類別: {LABEL_ENCODER.classes_.tolist()}")
        
        # 初始化 SHAP explainer（可選）
        try:
            # 使用小樣本初始化
            X_sample = pd.DataFrame(
                SCALER.mean_.reshape(1, -1),
                columns=FEATURES
            )
            EXPLAINER = shap.Explainer(MODEL.predict, X_sample)
            print("✅ SHAP explainer 初始化成功")
        except Exception as e:
            print(f"⚠️  SHAP explainer 初始化失敗: {e}")
            EXPLAINER = None
        
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        raise


def prepare_features(data: Dict) -> pd.DataFrame:
    """準備特徵"""
    # 建立完整特徵字典（含衍生特徵）
    df = pd.DataFrame([data])
    
    # 衍生特徵
    if 'koi_period' in df.columns and 'koi_duration' in df.columns:
        df['duration_period_ratio'] = df['koi_duration'] / (df['koi_period'] * 24)
    
    if 'koi_depth' in df.columns and 'koi_prad' in df.columns and df['koi_prad'].notna().any():
        df['depth_radius_ratio'] = df['koi_depth'] / (df['koi_prad'] ** 2 + 1e-6)
    
    if 'koi_model_snr' in df.columns and df['koi_model_snr'].notna().any():
        df['log_snr'] = np.log1p(df['koi_model_snr'])
    
    # 填補缺失值
    df = df.fillna(df.median())
    
    # 只保留模型需要的特徵
    available_features = [f for f in FEATURES if f in df.columns]
    missing_features = set(FEATURES) - set(available_features)
    
    if missing_features:
        print(f"⚠️  缺少特徵: {missing_features}")
        # 用 0 填補
        for feat in missing_features:
            df[feat] = 0
    
    return df[FEATURES]


@app.get("/")
async def root():
    """API 根路徑"""
    return {
        "name": "Exoplanet Detection API",
        "version": "1.0.0",
        "status": "online" if MODEL is not None else "model not loaded",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict-batch",
            "explain": "/explain",
            "metrics": "/metrics",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """健康檢查"""
    return {
        "status": "healthy" if MODEL is not None else "unhealthy",
        "model_loaded": MODEL is not None,
        "features_count": len(FEATURES) if FEATURES else 0,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: FeatureInput):
    """單筆預測"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="模型尚未載入")
    
    try:
        # 準備特徵
        X = prepare_features(input_data.dict())
        
        # 標準化
        X_scaled = SCALER.transform(X)
        
        # 預測
        proba = MODEL.predict_proba(X_scaled)[0]
        pred_idx = np.argmax(proba)
        pred_label = LABEL_ENCODER.inverse_transform([pred_idx])[0]
        
        # 建立機率字典
        prob_dict = {
            label: float(prob)
            for label, prob in zip(LABEL_ENCODER.classes_, proba)
        }
        
        return PredictionResponse(
            label=pred_label,
            confidence=float(proba[pred_idx]),
            probabilities=prob_dict,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"預測失敗: {str(e)}")


@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    """批次預測"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="模型尚未載入")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="僅支援 CSV 格式")
    
    try:
        # 讀取 CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        print(f"📊 批次預測: {len(df)} 筆資料")
        
        # 準備特徵
        results = []
        for idx, row in df.iterrows():
            try:
                X = prepare_features(row.to_dict())
                X_scaled = SCALER.transform(X)
                
                proba = MODEL.predict_proba(X_scaled)[0]
                pred_idx = np.argmax(proba)
                pred_label = LABEL_ENCODER.inverse_transform([pred_idx])[0]
                
                results.append({
                    'index': int(idx),
                    'label': pred_label,
                    'confidence': float(proba[pred_idx]),
                    'probabilities': {
                        label: float(prob)
                        for label, prob in zip(LABEL_ENCODER.classes_, proba)
                    }
                })
            except Exception as e:
                results.append({
                    'index': int(idx),
                    'error': str(e)
                })
        
        # 統計
        labels = [r['label'] for r in results if 'label' in r]
        summary = {label: labels.count(label) for label in set(labels)}
        
        return BatchPredictionResponse(
            total=len(results),
            predictions=results,
            summary=summary,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批次預測失敗: {str(e)}")


@app.post("/explain", response_model=ExplainResponse)
async def explain(input_data: FeatureInput):
    """解釋單筆預測"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="模型尚未載入")
    
    try:
        # 準備特徵
        X = prepare_features(input_data.dict())
        X_scaled = SCALER.transform(X)
        
        # 預測
        proba = MODEL.predict_proba(X_scaled)[0]
        pred_idx = np.argmax(proba)
        pred_label = LABEL_ENCODER.inverse_transform([pred_idx])[0]
        
        # 特徵重要度（簡化版，使用模型係數或特徵值）
        feature_importance = []
        for feat, val in zip(FEATURES, X_scaled[0]):
            feature_importance.append({
                'feature': feat,
                'value': float(val),
                'abs_value': float(abs(val))
            })
        
        # 排序
        feature_importance.sort(key=lambda x: x['abs_value'], reverse=True)
        top_features = feature_importance[:10]
        
        # SHAP values（如果可用）
        shap_values = []
        if EXPLAINER is not None:
            try:
                shap_vals = EXPLAINER(X_scaled)
                shap_values = shap_vals.values[0].tolist()
            except:
                pass
        
        return ExplainResponse(
            prediction=pred_label,
            top_features=top_features,
            shap_values=shap_values
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解釋失敗: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """獲取模型指標"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="模型尚未載入")
    
    return {
        "model_info": {
            "type": "Stacking Ensemble",
            "features_count": len(FEATURES),
            "classes": LABEL_ENCODER.classes_.tolist()
        },
        "features": FEATURES,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║     系外行星檢測 API 服務                              ║
    ║     Exoplanet Detection API Server                    ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000)