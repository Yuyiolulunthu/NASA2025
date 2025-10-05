"""
使用 NASA 真實資料訓練系外行星檢測模型 - 確定性版本
修正了所有隨機性來源，確保每次訓練結果一致
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime
import random
import os

from sklearn.model_selection import GroupKFold, cross_val_predict, StratifiedKFold
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


def set_all_seeds(seed=42):
    """設定所有隨機種子以確保結果可重現"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # TensorFlow (如果使用)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except:
        pass
    
    # PyTorch (如果使用)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except:
        pass
    
    print(f"All random seeds set to {seed}")


class NASAExoplanetTrainer:
    """NASA 真實資料訓練器 - 確定性版本"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.selected_features = None
        self.model = None
        
        Path("models").mkdir(exist_ok=True)
        
        # 設定全局隨機種子
        set_all_seeds(self.random_state)
    
    def load_tess_data(self, filepath='./tois.csv'):
        """載入 TESS 資料"""
        print(f"載入 TESS 資料: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"原始資料: {df.shape}")
        print(f"欄位名稱: {list(df.columns)}")
        
        # 使用 TESS Disposition 作為標籤
        if 'TESS Disposition' in df.columns:
            disp_col = 'TESS Disposition'
        elif 'TFOPWG Disposition' in df.columns:
            disp_col = 'TFOPWG Disposition'
        else:
            raise ValueError("找不到 TESS Disposition 或 TFOPWG Disposition 欄位")
        
        print(f"\n'{disp_col}' 欄位的實際值:")
        print(df[disp_col].value_counts())
        
        # 過濾掉空值
        df = df.dropna(subset=[disp_col])
        
        # 將類別合併成有意義的組
        label_mapping = {
            'KP': 'PLANET',      # Known Planet
            'CP': 'PLANET',      # Confirmed Planet
            'PC': 'PLANET',      # Planet Candidate
            'EB': 'FALSE_POSITIVE',  # Eclipsing Binary
            'FP': 'FALSE_POSITIVE',  # False Positive
            'IS': 'OTHER',       # Instrumental Signal
            'V': 'OTHER',        # Variable Star
            'O': 'OTHER'         # Other
        }
        
        df['disposition'] = df[disp_col].map(label_mapping)
        df = df.dropna(subset=['disposition'])
        
        print(f"\n過濾後資料: {df.shape}")
        print(f"\n合併後類別分佈:")
        print(df['disposition'].value_counts())
        print(f"\n類別說明:")
        print("  - PLANET: 行星（已知+確認+候選）")
        print("  - FALSE_POSITIVE: 假陽性（食雙星+假陽性）")
        print("  - OTHER: 其他（儀器訊號+變星+其他）")
        
        return df
    
    def engineer_features(self, df):
        """特徵工程 - 使用 TESS 真實欄位"""
        print("\n特徵工程...")
        
        # TESS 的實際欄位名稱
        feature_mapping = {
            'Period (days)': 'period',
            'Duration (hours)': 'duration',
            'Depth (ppm)': 'depth',
            'Planet Radius (R_Earth)': 'planet_radius',
            'Planet Equil Temp (K)': 'equil_temp',
            'Planet Insolation (Earth Flux)': 'insolation',
            'Planet SNR': 'snr',
            'Stellar Eff Temp (K)': 'stellar_temp',
            'Stellar Radius (R_Sun)': 'stellar_radius',
            'Stellar log(g) (cm/s^2)': 'stellar_logg',
            'Stellar Mass (M_Sun)': 'stellar_mass',
            'TESS Mag': 'tess_mag',
            'Stellar Distance (pc)': 'distance'
        }
        
        # 選擇存在的欄位
        available_features = {}
        for orig_name, new_name in feature_mapping.items():
            if orig_name in df.columns:
                available_features[orig_name] = new_name
        
        print(f"找到 {len(available_features)} 個可用特徵:")
        for orig, new in available_features.items():
            print(f"  - {orig} -> {new}")
        
        # 建立特徵矩陣
        X = df[list(available_features.keys())].copy()
        X.columns = list(available_features.values())
        
        # 處理缺失值 - 使用固定策略確保一致性
        print(f"\n缺失值處理:")
        for col in X.columns:
            missing_count = X[col].isnull().sum()
            if missing_count > 0:
                median_val = X[col].median()
                X[col].fillna(median_val, inplace=True)
                print(f"  填補 {col}: {missing_count} 個缺失值 (中位數={median_val:.2f})")
        
        # 衍生特徵
        if 'period' in X.columns and 'duration' in X.columns:
            X['duration_period_ratio'] = X['duration'] / (X['period'] * 24 + 1e-6)
        
        if 'depth' in X.columns and 'planet_radius' in X.columns:
            X['depth_radius_ratio'] = X['depth'] / (X['planet_radius'] ** 2 + 1e-6)
        
        if 'snr' in X.columns:
            X['log_snr'] = np.log1p(X['snr'])
        
        if 'equil_temp' in X.columns and 'insolation' in X.columns:
            X['temp_insol_ratio'] = X['equil_temp'] / (X['insolation'] + 1e-6)
        
        if 'stellar_temp' in X.columns and 'stellar_radius' in X.columns:
            X['stellar_luminosity'] = (X['stellar_radius'] ** 2) * ((X['stellar_temp'] / 5778) ** 4)
        
        # 移除無限值和極端值
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        print(f"\n特徵工程完成: {X.shape[1]} 個特徵")
        print(f"特徵列表: {X.columns.tolist()}")
        
        return X
    
    def select_features(self, X, y, threshold=50):
        """特徵選擇"""
        print(f"\n特徵選擇 (Top {threshold})...")
        
        # 先編碼標籤為整數
        le_temp = LabelEncoder()
        y_encoded = le_temp.fit_transform(np.array(y)).astype(np.int32)
        
        # 使用 LGBM 進行特徵選擇 - 確保 random_state
        lgbm = LGBMClassifier(
            n_estimators=100,
            random_state=self.random_state,  # 關鍵！
            verbose=-1,
            n_jobs=1,
            deterministic=True  # LightGBM 的確定性模式
        )
        lgbm.fit(X, y_encoded)
        
        importances = pd.Series(
            lgbm.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        
        # 選擇 Top N 或所有特徵
        n_select = min(threshold, len(X.columns))
        self.selected_features = importances.head(n_select).index.tolist()
        
        print(f"選擇了 {len(self.selected_features)} 個特徵")
        print(f"Top 10 重要特徵:")
        for i, (feat, imp) in enumerate(importances.head(10).items(), 1):
            print(f"  {i}. {feat}: {imp:.4f}")
        
        return X[self.selected_features]
    
    def build_model(self):
        """構建堆疊模型 - 確保所有組件都是確定性的"""
        print("\n構建堆疊模型...")
        
        # 所有基礎學習器都必須設定 random_state
        base_learners = [
            ('lgbm', LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=self.random_state,  # 關鍵！
                verbose=-1,
                n_jobs=1,
                deterministic=True  # LightGBM 確定性模式
            )),
            ('xgb', XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=self.random_state,  # 關鍵！
                eval_metric='logloss',
                verbosity=0,
                n_jobs=1
            )),
            ('catboost', CatBoostClassifier(
                iterations=200,
                learning_rate=0.05,
                depth=5,
                random_state=self.random_state,  # 關鍵！
                verbose=False,
                thread_count=1
            )),
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight='balanced',
                random_state=self.random_state,  # 關鍵！
                n_jobs=1
            ))
        ]
        
        # Meta learner 也要設定 random_state
        meta_learner = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=self.random_state  # 關鍵！
        )
        
        # StackingClassifier 的 cv 必須明確設定 random_state
        cv_strategy = StratifiedKFold(
            n_splits=3, 
            shuffle=True, 
            random_state=self.random_state  # 關鍵！
        )
        
        stacking_model = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta_learner,
            cv=cv_strategy,  # 使用固定的 CV 策略而不是整數
            n_jobs=1
        )
        
        print("模型構建完成 - 所有組件都已設定 random_state")
        return stacking_model
    
    def train(self, X, y, groups):
        """訓練模型"""
        print("\n" + "="*60)
        print("開始訓練")
        print("="*60)
        
        # 標準化
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # 編碼標籤
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"\n類別對應:")
        for i, label in enumerate(self.label_encoder.classes_):
            print(f"  {i}: {label}")
        
        # GroupKFold - 不需要 random_state (按組分割)
        gkf = GroupKFold(n_splits=5)
        
        # 構建模型
        stacking_model = self.build_model()
        
        # 交叉驗證
        print(f"\n5-Fold 交叉驗證...")
        oof_preds = cross_val_predict(
            stacking_model, X_scaled, y_encoded,
            cv=gkf.split(X_scaled, y_encoded, groups),
            method='predict_proba',
            n_jobs=1,
            verbose=1
        )
        
        # 完整訓練
        print("\n完整資料訓練...")
        stacking_model.fit(X_scaled, y_encoded)
        
        # 機率校準 - 確保 CV 也是確定性的
        print("\n機率校準...")
        cv_calibration = StratifiedKFold(
            n_splits=3, 
            shuffle=True, 
            random_state=self.random_state  # 關鍵！
        )
        
        calibrated_model = CalibratedClassifierCV(
            stacking_model,
            method='isotonic',
            cv=cv_calibration  # 使用固定的 CV 策略
        )
        calibrated_model.fit(X_scaled, y_encoded)
        
        self.model = calibrated_model
        
        # 評估
        self._evaluate(y_encoded, oof_preds)
        
        print("\n訓練完成！")
        return self
    
    def _evaluate(self, y_true, y_pred_proba):
        """評估模型"""
        print("\n" + "="*60)
        print("模型評估")
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
            'data_source': 'NASA TESS TOI',
            'random_state': self.random_state,  # 記錄隨機種子
            'is_deterministic': True  # 標記為確定性模型
        }
        
        model_path = Path("models") / f"exoplanet_model_{version}.pkl"
        joblib.dump(save_dict, model_path)
        
        feature_path = Path("models") / "feature_list.json"
        with open(feature_path, 'w') as f:
            json.dump({
                'features': self.selected_features,
                'version': version,
                'data_source': 'NASA TESS TOI',
                'random_state': self.random_state,
                'is_deterministic': True
            }, f, indent=2)
        
        print(f"\n模型已儲存: {model_path}")
        print(f"特徵列表: {feature_path}")
        print(f"\n模型資訊:")
        print(f"  - Random State: {self.random_state}")
        print(f"  - 確定性: True")
        print(f"  - 特徵數: {len(self.selected_features)}")
        
        return model_path


def main():
    """主訓練流程"""
    
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║   NASA Exoplanet Real Data Training (Deterministic)  ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    # 檢查資料檔案
    data_file = './tois.csv'
    if not Path(data_file).exists():
        print(f"找不到 {data_file}")
        print("\n請先下載 NASA 資料：")
        print("1. 執行: python download_nasa_data.py")
        print("2. 或手動下載: https://exoplanetarchive.ipac.caltech.edu/")
        return
    
    # 初始化訓練器 - 指定 random_state
    trainer = NASAExoplanetTrainer(random_state=42)
    
    # 載入資料
    df = trainer.load_tess_data(data_file)
    
    # 特徵工程
    X = trainer.engineer_features(df)
    y = df['disposition']
    
    # 使用 TIC ID 或 TOI 作為分組依據
    if 'TIC ID' in df.columns:
        groups = df['TIC ID']
    elif 'TOI' in df.columns:
        groups = df['TOI']
    else:
        groups = np.arange(len(df))
    
    # 特徵選擇
    X_selected = trainer.select_features(X, y, threshold=50)
    
    # 訓練
    trainer.train(X_selected, y, groups)
    
    # 儲存
    model_path = trainer.save()
    
    print("\n" + "="*60)
    print("訓練完成！")
    print("="*60)
    print("\n模型資訊:")
    print(f"  - 訓練資料: {len(df):,} 筆")
    print(f"  - 特徵數: {len(trainer.selected_features)}")
    print(f"  - 類別數: {len(trainer.label_encoder.classes_)}")
    print(f"  - 模型檔案: {model_path}")
    print(f"  - 確定性: True (random_state=42)")
    print("\n測試確定性:")
    print("  運行兩次訓練應該得到完全相同的結果")
    print("\n下一步:")
    print("1. 測試模型: python test_model.py")
    print("2. 啟動 API: python backend/app.py")
    print("3. 啟動前端: streamlit run Web/pages/analyze.py")


if __name__ == "__main__":
    main()