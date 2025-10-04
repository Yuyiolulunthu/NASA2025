"""
靈活的系外行星模型訓練器
支援多種模型配置和訓練策略
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import GroupKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings('ignore')


class ModelFactory:
    """模型工廠 - 可以建立各種不同的模型"""
    
    @staticmethod
    def get_model(model_type, **kwargs):
        """
        獲取指定類型的模型
        
        可用模型:
        - 'lgbm': LightGBM
        - 'xgb': XGBoost
        - 'catboost': CatBoost
        - 'random_forest': 隨機森林
        - 'gradient_boosting': 梯度提升
        - 'logistic': 邏輯迴歸
        - 'svm': 支持向量機
        - 'mlp': 神經網路
        - 'stacking': 堆疊模型（自動組合多個模型）
        """
        
        models = {
            'lgbm': lambda: LGBMClassifier(
                n_estimators=kwargs.get('n_estimators', 200),
                learning_rate=kwargs.get('learning_rate', 0.05),
                max_depth=kwargs.get('max_depth', 5),
                random_state=42,
                verbose=-1,
                n_jobs=1
            ),
            
            'xgb': lambda: XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 200),
                learning_rate=kwargs.get('learning_rate', 0.05),
                max_depth=kwargs.get('max_depth', 5),
                random_state=42,
                eval_metric='logloss',
                verbosity=0,
                n_jobs=1
            ),
            
            'catboost': lambda: CatBoostClassifier(
                iterations=kwargs.get('n_estimators', 200),
                learning_rate=kwargs.get('learning_rate', 0.05),
                depth=kwargs.get('max_depth', 5),
                random_state=42,
                verbose=False,
                thread_count=1
            ),
            
            'random_forest': lambda: RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', 10),
                class_weight='balanced',
                random_state=42,
                n_jobs=1
            ),
            
            'gradient_boosting': lambda: GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 200),
                learning_rate=kwargs.get('learning_rate', 0.05),
                max_depth=kwargs.get('max_depth', 5),
                random_state=42
            ),
            
            'logistic': lambda: LogisticRegression(
                max_iter=kwargs.get('max_iter', 1000),
                class_weight='balanced',
                random_state=42
            ),
            
            'svm': lambda: SVC(
                kernel=kwargs.get('kernel', 'rbf'),
                C=kwargs.get('C', 1.0),
                class_weight='balanced',
                probability=True,
                random_state=42
            ),
            
            'mlp': lambda: MLPClassifier(
                hidden_layer_sizes=kwargs.get('hidden_layers', (100, 50)),
                max_iter=kwargs.get('max_iter', 500),
                random_state=42
            ),
            
            'stacking': lambda: ModelFactory._build_stacking_model(**kwargs)
        }
        
        if model_type not in models:
            raise ValueError(f"未知的模型類型: {model_type}. 可用: {list(models.keys())}")
        
        return models[model_type]()
    
    @staticmethod
    def _build_stacking_model(**kwargs):
        """建立堆疊模型"""
        base_learners = [
            ('lgbm', LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
                verbose=-1,
                n_jobs=1
            )),
            ('xgb', XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
                eval_metric='logloss',
                verbosity=0,
                n_jobs=1
            )),
            ('catboost', CatBoostClassifier(
                iterations=200,
                learning_rate=0.05,
                depth=5,
                random_state=42,
                verbose=False,
                thread_count=1
            )),
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=1
            ))
        ]
        
        meta_learner = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        
        return StackingClassifier(
            estimators=base_learners,
            final_estimator=meta_learner,
            cv=3,
            n_jobs=1
        )


class FlexibleExoplanetTrainer:
    """靈活的系外行星訓練器"""
    
    def __init__(self, model_type='stacking', **model_params):
        """
        初始化訓練器
        
        Args:
            model_type: 模型類型 ('lgbm', 'xgb', 'random_forest', 'stacking', 等)
            **model_params: 模型的超參數
        """
        self.model_type = model_type
        self.model_params = model_params
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.selected_features = None
        self.model = None
        
        Path("models").mkdir(exist_ok=True)
        
        print(f"🎯 選擇模型: {model_type}")
        if model_params:
            print(f"📝 模型參數: {model_params}")
    
    def load_data(self, filepath='./tois.csv', label_strategy='binary'):
        """
        載入資料
        
        Args:
            filepath: 資料檔案路徑
            label_strategy: 標籤策略
                - 'binary': 二分類 (PLANET vs NOT_PLANET)
                - 'three_class': 三分類 (PLANET, FALSE_POSITIVE, OTHER)
                - 'full': 保留所有原始類別
        """
        print(f"\n📥 載入資料: {filepath}")
        print(f"🏷️  標籤策略: {label_strategy}")
        
        df = pd.read_csv(filepath)
        print(f"✓ 原始資料: {df.shape}")
        
        # 找到標籤欄位
        if 'TESS Disposition' in df.columns:
            disp_col = 'TESS Disposition'
        elif 'TFOPWG Disposition' in df.columns:
            disp_col = 'TFOPWG Disposition'
        else:
            raise ValueError("找不到標籤欄位")
        
        # 根據策略設定標籤
        df = df.dropna(subset=[disp_col])
        
        if label_strategy == 'binary':
            label_mapping = {
                'KP': 'PLANET', 'CP': 'PLANET', 'PC': 'PLANET',
                'EB': 'NOT_PLANET', 'FP': 'NOT_PLANET',
                'IS': 'NOT_PLANET', 'V': 'NOT_PLANET', 'O': 'NOT_PLANET'
            }
        elif label_strategy == 'three_class':
            label_mapping = {
                'KP': 'PLANET', 'CP': 'PLANET', 'PC': 'PLANET',
                'EB': 'FALSE_POSITIVE', 'FP': 'FALSE_POSITIVE',
                'IS': 'OTHER', 'V': 'OTHER', 'O': 'OTHER'
            }
        else:  # full
            label_mapping = None
            df['disposition'] = df[disp_col]
        
        if label_mapping:
            df['disposition'] = df[disp_col].map(label_mapping)
            df = df.dropna(subset=['disposition'])
        
        print(f"✓ 過濾後資料: {df.shape}")
        print(f"\n類別分佈:")
        print(df['disposition'].value_counts())
        
        return df
    
    def engineer_features(self, df):
        """特徵工程"""
        print("\n🔧 特徵工程...")
        
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
        
        available_features = {k: v for k, v in feature_mapping.items() if k in df.columns}
        X = df[list(available_features.keys())].copy()
        X.columns = list(available_features.values())
        
        # 處理缺失值
        for col in X.columns:
            if X[col].isnull().any():
                X[col].fillna(X[col].median(), inplace=True)
        
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
        
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        print(f"✓ 特徵工程完成: {X.shape[1]} 個特徵")
        
        # 儲存所有特徵（不做特徵選擇，讓模型自己學習）
        self.selected_features = X.columns.tolist()
        
        return X
    
    def train(self, X, y, groups=None, use_cross_validation=True):
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
        
        # 建立模型
        print(f"\n🏗️  建立模型: {self.model_type}")
        model = ModelFactory.get_model(self.model_type, **self.model_params)
        
        if use_cross_validation and groups is not None:
            # 交叉驗證
            print(f"\n📊 5-Fold 交叉驗證...")
            gkf = GroupKFold(n_splits=5)
            
            oof_preds = cross_val_predict(
                model, X_scaled, y_encoded,
                cv=gkf.split(X_scaled, y_encoded, groups),
                method='predict_proba',
                n_jobs=1,
                verbose=1
            )
            
            # 評估交叉驗證結果
            self._evaluate(y_encoded, oof_preds, "交叉驗證")
        
        # 完整訓練
        print("\n🎯 完整資料訓練...")
        model.fit(X_scaled, y_encoded)
        
        # 機率校準 - 檢查每個類別的樣本數
        print("\n⚖️  機率校準...")
        min_samples = min(np.bincount(y_encoded))
        
        if min_samples >= 3:
            # 動態調整 cv 值
            cv_folds = min(3, min_samples)
            print(f"   使用 {cv_folds}-fold 交叉驗證")
            
            calibrated_model = CalibratedClassifierCV(
                model,
                method='isotonic',
                cv=cv_folds
            )
            calibrated_model.fit(X_scaled, y_encoded)
            self.model = calibrated_model
        else:
            print(f"   ⚠️  某些類別樣本數太少 (最少={min_samples})，跳過機率校準")
            self.model = model
        
        # 在訓練集上評估
        train_preds = self.model.predict_proba(X_scaled)
        self._evaluate(y_encoded, train_preds, "訓練集")
        
        print("\n✅ 訓練完成！")
        return self
    
    def _evaluate(self, y_true, y_pred_proba, dataset_name=""):
        """評估模型"""
        print("\n" + "="*60)
        print(f"📈 模型評估 - {dataset_name}")
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
            'model_type': self.model_type,
            'model_params': self.model_params,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'data_source': 'NASA TESS TOI'
        }
        
        model_name = f"exoplanet_{self.model_type}_{version}.pkl"
        model_path = Path("models") / model_name
        joblib.dump(save_dict, model_path)
        
        print(f"\n💾 模型已儲存: {model_path}")
        
        return model_path


def main():
    """主訓練流程"""
    
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║     Flexible Exoplanet Model Training                 ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    print("可用的模型:")
    print("1. LightGBM (lgbm)")
    print("2. XGBoost (xgb)")
    print("3. CatBoost (catboost)")
    print("4. Random Forest (random_forest)")
    print("5. Gradient Boosting (gradient_boosting)")
    print("6. Logistic Regression (logistic)")
    print("7. SVM (svm)")
    print("8. Neural Network (mlp)")
    print("9. Stacking Ensemble (stacking)")
    
    model_choice = input("\n請選擇模型 (1-9, 預設=9): ").strip()
    
    model_map = {
        '1': 'lgbm', '2': 'xgb', '3': 'catboost',
        '4': 'random_forest', '5': 'gradient_boosting',
        '6': 'logistic', '7': 'svm', '8': 'mlp', '9': 'stacking'
    }
    
    model_type = model_map.get(model_choice, 'stacking')
    
    print("\n標籤策略:")
    print("1. 二分類 (PLANET vs NOT_PLANET)")
    print("2. 三分類 (PLANET, FALSE_POSITIVE, OTHER)")
    print("3. 完整分類 (保留所有原始類別)")
    
    label_choice = input("\n請選擇標籤策略 (1-3, 預設=2): ").strip()
    
    label_map = {
        '1': 'binary',
        '2': 'three_class',
        '3': 'full'
    }
    
    label_strategy = label_map.get(label_choice, 'three_class')
    
    # 模型參數（可以根據需要調整）
    model_params = {}
    
    if model_type in ['lgbm', 'xgb', 'catboost', 'random_forest', 'gradient_boosting']:
        n_est = input("\n樹的數量 (預設=200): ").strip()
        if n_est:
            model_params['n_estimators'] = int(n_est)
    
    # 初始化訓練器
    trainer = FlexibleExoplanetTrainer(model_type, **model_params)
    
    # 載入資料
    data_file = './tois.csv'
    df = trainer.load_data(data_file, label_strategy=label_strategy)
    
    # 特徵工程
    X = trainer.engineer_features(df)
    y = df['disposition']
    groups = df['TIC ID'] if 'TIC ID' in df.columns else None
    
    # 訓練
    trainer.train(X, y, groups)
    
    # 儲存
    model_path = trainer.save()
    
    print("\n" + "="*60)
    print("🎉 訓練完成！")
    print("="*60)
    print(f"\n模型資訊:")
    print(f"  - 模型類型: {model_type}")
    print(f"  - 訓練資料: {len(df):,} 筆")
    print(f"  - 特徵數: {len(trainer.selected_features)}")
    print(f"  - 類別數: {len(trainer.label_encoder.classes_)}")
    print(f"  - 模型檔案: {model_path}")


if __name__ == "__main__":
    main()