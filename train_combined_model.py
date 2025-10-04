"""
合併多個資料集訓練系外行星檢測模型
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

from unified_data_loader import UnifiedDataLoader
from train_custom_model import FlexibleExoplanetTrainer, ModelFactory

import warnings
warnings.filterwarnings('ignore')


class CombinedDatasetTrainer:
    """合併資料集訓練器"""
    
    def __init__(self, model_type='stacking', **model_params):
        self.loader = UnifiedDataLoader()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model_type = model_type
        self.model_params = model_params
        self.model = None
        self.feature_cols = None
        self.datasets_used = []
        
        Path("models").mkdir(exist_ok=True)
    
    def load_multiple_datasets(self, file_configs, label_strategy='binary'):
        """
        載入並合併多個資料集
        
        Args:
            file_configs: list of dict, 例如:
                [
                    {'path': './tois.csv', 'type': 'tess'},
                    {'path': './data/kepler_koi.csv', 'type': 'kepler'}
                ]
            label_strategy: 'binary' or 'three_class'
        
        Returns:
            合併後的 X, y, dataset_sources
        """
        print("\n" + "="*70)
        print("載入並合併多個資料集")
        print("="*70)
        
        all_X = []
        all_y = []
        all_sources = []
        
        for config in file_configs:
            filepath = config['path']
            dataset_type = config.get('type', None)
            
            print(f"\n處理資料集: {filepath}")
            
            # 載入資料
            df, detected_type = self.loader.load_dataset(
                filepath, dataset_type, label_strategy
            )
            
            # 提取特徵和標籤
            X, y, groups = self.loader.get_features_and_labels(df, detected_type)
            
            # 記錄資料來源
            sources = [detected_type] * len(X)
            
            all_X.append(X)
            all_y.append(y)
            all_sources.extend(sources)
            
            self.datasets_used.append({
                'name': detected_type,
                'path': filepath,
                'samples': len(X)
            })
        
        # 合併資料
        print("\n" + "="*70)
        print("合併資料集")
        print("="*70)
        
        # 確保所有資料集有相同的特徵
        all_features = set()
        for X in all_X:
            all_features.update(X.columns)
        
        self.feature_cols = sorted(list(all_features))
        
        # 補齊缺失特徵
        aligned_X = []
        for X in all_X:
            X_aligned = X.copy()
            for feat in self.feature_cols:
                if feat not in X_aligned.columns:
                    X_aligned[feat] = 0  # 缺失特徵用0填充
            X_aligned = X_aligned[self.feature_cols]
            aligned_X.append(X_aligned)
        
        # 合併
        X_combined = pd.concat(aligned_X, axis=0, ignore_index=True)
        y_combined = pd.concat(all_y, axis=0, ignore_index=True)
        
        print(f"\n合併後資料:")
        print(f"  總樣本數: {len(X_combined):,}")
        print(f"  特徵數: {len(self.feature_cols)}")
        print(f"\n各資料集貢獻:")
        for dataset_info in self.datasets_used:
            pct = (dataset_info['samples'] / len(X_combined)) * 100
            print(f"  {dataset_info['name']:15s}: {dataset_info['samples']:5,} ({pct:5.1f}%)")
        
        print(f"\n合併後類別分佈:")
        print(y_combined.value_counts())
        
        return X_combined, y_combined, all_sources
    
    def train(self, X, y, test_size=0.2):
        """訓練合併模型"""
        print("\n" + "="*70)
        print("訓練合併模型")
        print("="*70)
        
        # 分割訓練集和測試集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\n資料分割:")
        print(f"  訓練集: {len(X_train):,} ({(1-test_size)*100:.0f}%)")
        print(f"  測試集: {len(X_test):,} ({test_size*100:.0f}%)")
        
        # 標準化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        # 編碼標籤
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        print(f"\n類別對應:")
        for i, label in enumerate(self.label_encoder.classes_):
            count_train = (y_train_encoded == i).sum()
            count_test = (y_test_encoded == i).sum()
            print(f"  {i}: {label:20s} (訓練: {count_train:,}, 測試: {count_test:,})")
        
        # 建立模型
        print(f"\n建立模型: {self.model_type}")
        self.model = ModelFactory.get_model(self.model_type, **self.model_params)
        
        # 交叉驗證
        print("\n進行 5-Fold 交叉驗證...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train_encoded,
            cv=cv, scoring='accuracy', n_jobs=1
        )
        
        print(f"交叉驗證準確率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # 訓練最終模型
        print("\n訓練最終模型...")
        self.model.fit(X_train_scaled, y_train_encoded)
        
        # 評估
        print("\n" + "="*70)
        print("模型評估")
        print("="*70)
        
        # 訓練集
        train_preds = self.model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train_encoded, train_preds)
        print(f"\n訓練集準確率: {train_acc:.4f}")
        
        # 測試集
        test_preds = self.model.predict(X_test_scaled)
        test_probs = self.model.predict_proba(X_test_scaled)
        test_acc = accuracy_score(y_test_encoded, test_preds)
        
        print(f"測試集準確率: {test_acc:.4f}")
        
        print("\n測試集分類報告:")
        print(classification_report(
            y_test_encoded, test_preds,
            target_names=self.label_encoder.classes_
        ))
        
        print("\n混淆矩陣:")
        cm = confusion_matrix(y_test_encoded, test_preds)
        print(cm)
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(
                y_test_encoded, test_probs,
                multi_class='ovr',
                average='weighted'
            )
            print(f"\nWeighted ROC-AUC: {roc_auc:.4f}")
        except:
            pass
        
        return self
    
    def save(self, model_name='combined_model'):
        """儲存合併模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_dict = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_cols': self.feature_cols,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'datasets_used': self.datasets_used,
            'version': timestamp,
            'timestamp': datetime.now().isoformat(),
            'data_source': 'Combined Dataset'
        }
        
        model_path = Path("models") / f"{model_name}_{timestamp}.pkl"
        joblib.dump(save_dict, model_path)
        
        # 儲存摘要
        summary = {
            'model_name': model_name,
            'model_type': self.model_type,
            'version': timestamp,
            'feature_count': len(self.feature_cols),
            'class_count': len(self.label_encoder.classes_),
            'classes': list(self.label_encoder.classes_),
            'datasets_used': self.datasets_used,
            'total_samples': sum(d['samples'] for d in self.datasets_used)
        }
        
        summary_path = Path("models") / f"{model_name}_{timestamp}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n模型已儲存: {model_path}")
        print(f"摘要已儲存: {summary_path}")
        
        return model_path


def main():
    """主訓練流程"""
    
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║     Combined Dataset Training                         ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    # 設定要合併的資料集
    print("\n資料集配置:")
    print("="*70)
    
    datasets = []
    
    # TESS 資料
    tess_path = input("TESS 資料路徑 (預設=./tois.csv, 留空跳過): ").strip()
    if not tess_path:
        tess_path = './tois.csv'
    
    if Path(tess_path).exists():
        datasets.append({'path': tess_path, 'type': 'tess'})
        print(f"已加入 TESS 資料: {tess_path}")
    else:
        print(f"找不到檔案: {tess_path}")
    
    # Kepler 資料
    kepler_path = input("Kepler 資料路徑 (預設=./data/kepler_koi.csv, 留空跳過): ").strip()
    if not kepler_path:
        kepler_path = './data/kepler_koi.csv'
    
    if Path(kepler_path).exists():
        datasets.append({'path': kepler_path, 'type': 'kepler'})
        print(f"已加入 Kepler 資料: {kepler_path}")
    else:
        print(f"找不到檔案: {kepler_path}")
    
    # 其他資料集
    while True:
        other_path = input("其他資料路徑 (留空結束): ").strip()
        if not other_path:
            break
        if Path(other_path).exists():
            datasets.append({'path': other_path, 'type': None})
            print(f"已加入: {other_path}")
        else:
            print(f"找不到檔案: {other_path}")
    
    if not datasets:
        print("\n未指定任何資料集，結束")
        return
    
    # 選擇標籤策略
    print("\n標籤策略:")
    print("1. 二分類 (PLANET vs NOT_PLANET) - 推薦")
    print("2. 三分類 (PLANET, FALSE_POSITIVE, OTHER)")
    
    label_choice = input("\n請選擇 (1-2, 預設=1): ").strip()
    label_strategy = 'binary' if label_choice != '2' else 'three_class'
    
    # 選擇模型
    print("\n模型類型:")
    print("1. Stacking Ensemble (最準確，較慢)")
    print("2. LightGBM (快速)")
    print("3. XGBoost (平衡)")
    print("4. Random Forest (穩定)")
    
    model_choice = input("\n請選擇 (1-4, 預設=1): ").strip()
    model_map = {
        '1': 'stacking',
        '2': 'lgbm',
        '3': 'xgb',
        '4': 'random_forest'
    }
    model_type = model_map.get(model_choice, 'stacking')
    
    # 模型參數
    model_params = {}
    if model_type in ['lgbm', 'xgb', 'random_forest']:
        n_est = input(f"\n樹的數量 (預設=300): ").strip()
        if n_est:
            model_params['n_estimators'] = int(n_est)
        else:
            model_params['n_estimators'] = 300
    
    # 初始化訓練器
    trainer = CombinedDatasetTrainer(model_type, **model_params)
    
    # 載入並合併資料
    X, y, sources = trainer.load_multiple_datasets(datasets, label_strategy)
    
    # 訓練
    trainer.train(X, y, test_size=0.2)
    
    # 儲存
    model_name = input("\n模型名稱 (預設=combined_model): ").strip()
    if not model_name:
        model_name = 'combined_model'
    
    model_path = trainer.save(model_name)
    
    print("\n" + "="*70)
    print("訓練完成")
    print("="*70)
    print(f"\n模型資訊:")
    print(f"  模型類型: {model_type}")
    print(f"  資料集數: {len(datasets)}")
    print(f"  總樣本數: {len(X):,}")
    print(f"  特徵數: {len(trainer.feature_cols)}")
    print(f"  類別數: {len(trainer.label_encoder.classes_)}")
    print(f"  模型檔案: {model_path}")
    
    print("\n下一步:")
    print("  使用 test_model.py 測試此模型")
    print("  或用新的觀測資料進行預測")


if __name__ == "__main__":
    main()