
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')


class NASAExoplanetTrainer:
    """NASA 真實資料訓練器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.selected_features = None
        self.model = None
        
        Path("models").mkdir(exist_ok=True)
    
    def load_kepler_data(self, filepath='kepler_data.csv'):
        """載入 Kepler 資料"""
        print(f"📥 載入 Kepler 資料: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"✓ 原始資料: {df.shape}")
        
        # 標準化標籤名稱
        label_mapping = {
            'CONFIRMED': 'CONFIRMED',
            'CANDIDATE': 'CANDIDATE',
            'FALSE POSITIVE': 'FALSE POSITIVE',
            'NOT DISPOSITIONED': None  # 移除未分類的
        }
        
        df['disposition'] = df['koi_disposition'].map(label_mapping)
        df = df.dropna(subset=['disposition'])
        
        print(f"✓ 過濾後資料: {df.shape}")
        print(f"\n類別分佈:")
        print(df['disposition'].value_counts())
        
        return df
    
    def engineer_features(self, df):
        """特徵工程 - 使用 Kepler 真實欄位"""
        print("\n🔧 特徵工程...")
        
        # Kepler 的主要特徵欄位
        feature_columns = [
            'koi_period',       # 軌道週期
            'koi_duration',     # 凌日持續時間
            'koi_depth',        # 凌日深度
            'koi_prad',         # 行星半徑
            'koi_teq',          # 平衡溫度
            'koi_insol',        # 恆星輻射
            'koi_model_snr',    # 信噪比
            'koi_steff',        # 恆星溫度
            'koi_srad',         # 恆星半徑
            'koi_slogg',        # 恆星表面重力
        ]
        
        # 只保留存在的欄位
        available_features = [col for col in feature_columns if col in df.columns]
        
        X = df[available_features].copy()
        
        # 處理缺失值
        print(f"缺失值處理前: {X.shape}")
        
        # 用中位數填補數值型欄位
        for col in X.columns:
            if X[col].isnull().any():
                median_val = X[col].median()
                X[col].fillna(median_val, inplace=True)
                print(f"  填補 {col}: {X[col].isnull().sum()} 個缺失值")
        
        # 衍生特徵
        if 'koi_period' in X.columns and 'koi_duration' in X.columns:
            X['duration_period_ratio'] = X['koi_duration'] / (X['koi_period'] * 24 + 1e-6)
        
        if 'koi_depth' in X.columns and 'koi_prad' in X.columns:
            X['depth_radius_ratio'] = X['koi_depth'] / (X['koi_prad'] ** 2 + 1e-6)
        
        if 'koi_model_snr' in X.columns:
            X['log_snr'] = np.log1p(X['koi_model_snr'])
        
        if 'koi_teq' in X.columns and 'koi_insol' in X.columns:
            X['temp_insol_ratio'] = X['koi_teq'] / (X['koi_insol'] + 1e-6)
        
        # 移除無限值和極端值
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        print(f"✓ 特徵工程完成: {X.shape[1]} 個特徵")
        print(f"特徵列表: {X.columns.tolist()}")
        
        return X
    
    def select_features(self, X, y, threshold=50):
        """特徵選擇"""
        print(f"\n🎯 特徵選擇 (Top {threshold})...")
        
        lgbm = LGBMClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            verbose=-1
        )
        lgbm.fit(X, y)
        
        importances = pd.Series(
            lgbm.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        
        # 選擇 Top N 或所有特徵（如果特徵數 < threshold）
        n_select = min(threshold, len(X.columns))
        self.selected_features = importances.head(n_select).index.tolist()
        
        print(f"✓ 選擇了 {len(self.selected_features)} 個特徵")
        print(f"Top 10 重要特徵:")
        for i, (feat, imp) in enumerate(importances.head(10).items(), 1):
            print(f"  {i}. {feat}: {imp:.4f}")
        
        return X[self.selected_features]
    
    def build_model(self):
        """構建堆疊模型"""
        print("\n🏗️  構建堆疊模型...")
        
        base_learners = [
            ('lgbm', LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                class_weight='balanced',
                random_state=42,
                verbose=-1
            )),
            ('xgb', XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
                eval_metric='logloss',
                verbosity=0
            )),
            ('catboost', CatBoostClassifier(
                iterations=200,
                learning_rate=0.05,
                depth=5,
                random_state=42,
                verbose=False
            )),
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ))
        ]
        
        meta_learner = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        
        stacking_model = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta_learner,
            cv=3,
            n_jobs=-1
        )
        
        print("✓ 模型構建完成")
        return stacking_model
    
    def train(self, X, y, groups):
        """訓練模型"""
        print("\n" + "="*60)
        print("🚀 開始訓練")
        print("="*60)
        
        # 標準化
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # 編碼標籤
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"\n類別對應:")
        for i, label in enumerate(self.label_encoder.classes_):
            print(f"  {i}: {label}")
        
        # GroupKFold
        gkf = GroupKFold(n_splits=5)
        
        # 構建模型
        stacking_model = self.build_model()
        
        # 交叉驗證
        print(f"\n📊 5-Fold 交叉驗證...")
        oof_preds = cross_val_predict(
            stacking_model, X_scaled, y_encoded,
            cv=gkf.split(X_scaled, y_encoded, groups),
            method='predict_proba',
            n_jobs=-1,
            verbose=1
        )
        
        # 完整訓練
        print("\n🎯 完整資料訓練...")
        stacking_model.fit(X_scaled, y_encoded)
        
        # 機率校準
        print("\n⚖️  機率校準...")
        calibrated_model = CalibratedClassifierCV(
            stacking_model,
            method='isotonic',
            cv=3
        )
        calibrated_model.fit(X_scaled, y_encoded)
        
        self.model = calibrated_model
        
        # 評估
        self._evaluate(y_encoded, oof_preds)
        
        print("\n✅ 訓練完成！")
        return self
    
    def _evaluate(self, y_true, y_pred_proba):
        """評估模型"""
        print("\n" + "="*60)
        print("📈 模型評估")
        print("="*60)
        
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        print("\n分類報告:")
        print(classification_report(
            y_true, y_pred,
            target_names=self.label_encoder.classes_
        ))
        
        print("\n混淆矩陣:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        
        # 計算每個類別的準確率
        print("\n各類別準確率:")
        for i, label in enumerate(self.label_encoder.classes_):
            if cm[i].sum() > 0:
                acc = cm[i, i] / cm[i].sum()
                print(f"  {label}: {acc:.2%}")
        
        try:
            roc_auc = roc_auc_score(
                y_true, y_pred_proba,
                multi_class='ovr',
                average='weighted'
            )
            print(f"\nWeighted ROC-AUC: {roc_auc:.4f}")
        except:
            pass
    
    def save(self, version=None):
        """儲存模型"""
        version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_dict = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'selected_features': self.selected_features,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'data_source': 'NASA Kepler'
        }
        
        model_path = Path("models") / f"exoplanet_model_{version}.pkl"
        joblib.dump(save_dict, model_path)
        
        feature_path = Path("models") / "feature_list.json"
        with open(feature_path, 'w') as f:
            json.dump({
                'features': self.selected_features,
                'version': version,
                'data_source': 'NASA Kepler'
            }, f, indent=2)
        
        print(f"\n💾 模型已儲存: {model_path}")
        print(f"💾 特徵列表: {feature_path}")
        
        return model_path


def main():
    """主訓練流程"""
    
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║     NASA 真實資料訓練系統                              ║
    ║     NASA Exoplanet Real Data Training                 ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    # 檢查資料檔案
    data_file = 'kepler_data.csv'
    if not Path(data_file).exists():
        print(f"❌ 找不到 {data_file}")
        print("\n請先下載 NASA 資料：")
        print("1. 執行: python download_nasa_data.py")
        print("2. 或手動下載: https://exoplanetarchive.ipac.caltech.edu/")
        return
    
    # 初始化訓練器
    trainer = NASAExoplanetTrainer()
    
    # 載入資料
    df = trainer.load_kepler_data(data_file)
    
    # 特徵工程
    X = trainer.engineer_features(df)
    y = df['disposition']
    groups = df['kepid'] if 'kepid' in df.columns else np.arange(len(df))
    
    # 特徵選擇
    X_selected = trainer.select_features(X, y, threshold=50)
    
    # 訓練
    trainer.train(X_selected, y, groups)
    
    # 儲存
    model_path = trainer.save()
    
    print("\n" + "="*60)
    print("🎉 訓練完成！")
    print("="*60)
    print("\n模型資訊:")
    print(f"  - 訓練資料: {len(df):,} 筆")
    print(f"  - 特徵數: {len(trainer.selected_features)}")
    print(f"  - 類別數: {len(trainer.label_encoder.classes_)}")
    print(f"  - 模型檔案: {model_path}")
    print("\n下一步:")
    print("1. 啟動 API: python backend/app.py")
    print("2. 啟動前端: streamlit run frontend/simple_app.py")


if __name__ == "__main__":
    main()