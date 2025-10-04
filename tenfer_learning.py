"""
跨資料集遷移學習
在一個資料集上訓練，在另一個資料集上測試或微調
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from unified_data_loader import UnifiedDataLoader
from train_custom_model import FlexibleExoplanetTrainer, ModelFactory

import warnings
warnings.filterwarnings('ignore')


class TransferLearningPipeline:
    """遷移學習流程"""
    
    def __init__(self):
        self.loader = UnifiedDataLoader()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.source_model = None
        self.target_model = None
        self.feature_cols = None
    
    def load_source_data(self, filepath, dataset_type=None, label_strategy='binary'):
        """載入源資料集（用於訓練）"""
        print("\n" + "="*70)
        print("📚 載入源資料集 (Source Dataset)")
        print("="*70)
        
        df, detected_type = self.loader.load_dataset(filepath, dataset_type, label_strategy)
        X, y, groups = self.loader.get_features_and_labels(df, detected_type)
        
        self.source_dataset_type = detected_type
        self.feature_cols = X.columns.tolist()
        
        print(f"\n✓ 源資料集: {detected_type}")
        print(f"✓ 資料形狀: {X.shape}")
        print(f"✓ 類別分佈:\n{y.value_counts()}")
        
        return X, y, groups
    
    def load_target_data(self, filepath, dataset_type=None, label_strategy='binary'):
        """載入目標資料集（用於測試或微調）"""
        print("\n" + "="*70)
        print("🎯 載入目標資料集 (Target Dataset)")
        print("="*70)
        
        df, detected_type = self.loader.load_dataset(filepath, dataset_type, label_strategy)
        X, y, groups = self.loader.get_features_and_labels(df, detected_type)
        
        # 確保目標資料集使用相同的特徵
        missing_features = set(self.feature_cols) - set(X.columns)
        if missing_features:
            print(f"\n⚠️  目標資料集缺少以下特徵: {missing_features}")
            print("   將用 0 填充缺失特徵")
            for feat in missing_features:
                X[feat] = 0
        
        # 確保特徵順序一致
        X = X[self.feature_cols]
        
        self.target_dataset_type = detected_type
        
        print(f"\n✓ 目標資料集: {detected_type}")
        print(f"✓ 資料形狀: {X.shape}")
        if y is not None:
            print(f"✓ 類別分佈:\n{y.value_counts()}")
        
        return X, y, groups
    
    def train_source_model(self, X_source, y_source, model_type='lgbm', **model_params):
        """在源資料集上訓練模型"""
        print("\n" + "="*70)
        print("🚀 在源資料集上訓練模型")
        print("="*70)
        
        # 標準化
        X_scaled = self.scaler.fit_transform(X_source)
        X_scaled = pd.DataFrame(X_scaled, columns=X_source.columns)
        
        # 編碼標籤
        y_encoded = self.label_encoder.fit_transform(y_source)
        
        print(f"\n類別對應:")
        for i, label in enumerate(self.label_encoder.classes_):
            print(f"  {i}: {label}")
        
        # 建立並訓練模型
        print(f"\n🏗️  建立模型: {model_type}")
        self.source_model = ModelFactory.get_model(model_type, **model_params)
        
        print("\n📊 訓練中...")
        self.source_model.fit(X_scaled, y_encoded)
        
        # 評估源資料集性能
        train_preds = self.source_model.predict(X_scaled)
        train_acc = accuracy_score(y_encoded, train_preds)
        
        print(f"\n✓ 源資料集訓練完成")
        print(f"✓ 訓練準確率: {train_acc:.4f}")
        
        print("\n訓練集分類報告:")
        print(classification_report(
            y_encoded, train_preds,
            target_names=self.label_encoder.classes_
        ))
        
        return self.source_model
    
    def test_on_target(self, X_target, y_target):
        """直接在目標資料集上測試（零樣本遷移）"""
        print("\n" + "="*70)
        print("🔬 零樣本遷移測試 (Zero-shot Transfer)")
        print("="*70)
        print("說明: 直接用源模型預測目標資料，不做任何微調")
        
        # 標準化目標資料
        X_scaled = self.scaler.transform(X_target)
        X_scaled = pd.DataFrame(X_scaled, columns=X_target.columns)
        
        # 編碼目標標籤
        y_encoded = self.label_encoder.transform(y_target)
        
        # 預測
        preds = self.source_model.predict(X_scaled)
        probs = self.source_model.predict_proba(X_scaled)
        
        # 評估
        acc = accuracy_score(y_encoded, preds)
        
        print(f"\n目標資料集性能:")
        print(f"準確率: {acc:.4f}")
        
        print("\n分類報告:")
        print(classification_report(
            y_encoded, preds,
            target_names=self.label_encoder.classes_
        ))
        
        print("\n混淆矩陣:")
        cm = confusion_matrix(y_encoded, preds)
        print(cm)
        
        return acc, preds, probs
    
    def fine_tune_on_target(self, X_target, y_target, model_type='lgbm', fine_tune_ratio=0.2, **model_params):
        """在目標資料集上微調（遷移學習）"""
        print("\n" + "="*70)
        print("🎨 遷移學習微調 (Fine-tuning)")
        print("="*70)
        print(f"說明: 用目標資料集的 {fine_tune_ratio*100:.0f}% 進行微調，剩餘部分測試")
        
        # 分割目標資料集
        X_train, X_test, y_train, y_test = train_test_split(
            X_target, y_target, 
            test_size=1-fine_tune_ratio,
            random_state=42,
            stratify=y_target
        )
        
        print(f"\n資料分割:")
        print(f"  微調集: {len(X_train)} 筆")
        print(f"  測試集: {len(X_test)} 筆")
        
        # 標準化
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        # 編碼標籤
        y_train_encoded = self.label_encoder.transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # 微調模型
        print(f"\n🔧 微調模型...")
        self.target_model = ModelFactory.get_model(model_type, **model_params)
        
        # 選項1: 從頭訓練（使用較少資料）
        self.target_model.fit(X_train_scaled, y_train_encoded)
        
        # 評估
        train_preds = self.target_model.predict(X_train_scaled)
        test_preds = self.target_model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train_encoded, train_preds)
        test_acc = accuracy_score(y_test_encoded, test_preds)
        
        print(f"\n✓ 微調完成")
        print(f"✓ 微調集準確率: {train_acc:.4f}")
        print(f"✓ 測試集準確率: {test_acc:.4f}")
        
        print("\n測試集分類報告:")
        print(classification_report(
            y_test_encoded, test_preds,
            target_names=self.label_encoder.classes_
        ))
        
        print("\n混淆矩陣:")
        cm = confusion_matrix(y_test_encoded, test_preds)
        print(cm)
        
        return test_acc, self.target_model
    
    def compare_strategies(self, X_target, y_target, model_type='lgbm', **model_params):
        """比較不同遷移策略的效果"""
        print("\n" + "="*70)
        print("📊 遷移學習策略比較")
        print("="*70)
        
        results = {}
        
        # 策略1: 零樣本遷移
        print("\n【策略1】零樣本遷移（不使用目標資料訓練）")
        zero_shot_acc, _, _ = self.test_on_target(X_target, y_target)
        results['zero_shot'] = zero_shot_acc
        
        # 策略2: 用10%目標資料微調
        print("\n【策略2】用 10% 目標資料微調")
        fine_tune_10_acc, _ = self.fine_tune_on_target(
            X_target, y_target, model_type, fine_tune_ratio=0.1, **model_params
        )
        results['fine_tune_10'] = fine_tune_10_acc
        
        # 策略3: 用30%目標資料微調
        print("\n【策略3】用 30% 目標資料微調")
        fine_tune_30_acc, _ = self.fine_tune_on_target(
            X_target, y_target, model_type, fine_tune_ratio=0.3, **model_params
        )
        results['fine_tune_30'] = fine_tune_30_acc
        
        # 策略4: 完全在目標資料上訓練（基準）
        print("\n【策略4】完全在目標資料上訓練（基準）")
        X_train, X_test, y_train, y_test = train_test_split(
            X_target, y_target, test_size=0.3, random_state=42, stratify=y_target
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        target_only_model = ModelFactory.get_model(model_type, **model_params)
        target_only_model.fit(X_train_scaled, y_train_encoded)
        
        test_preds = target_only_model.predict(X_test_scaled)
        target_only_acc = accuracy_score(y_test_encoded, test_preds)
        results['target_only'] = target_only_acc
        
        print(f"\n✓ 測試集準確率: {target_only_acc:.4f}")
        
        # 總結比較
        print("\n" + "="*70)
        print("📈 遷移學習效果總結")
        print("="*70)
        
        print(f"\n策略                           準確率      提升")
        print("-" * 60)
        baseline = results['target_only']
        for strategy, acc in results.items():
            improvement = acc - baseline
            improvement_pct = (improvement / baseline) * 100 if baseline > 0 else 0
            
            strategy_names = {
                'zero_shot': '零樣本遷移 (0% 目標資料)',
                'fine_tune_10': '微調 (10% 目標資料)',
                'fine_tune_30': '微調 (30% 目標資料)',
                'target_only': '完全目標訓練 (70% 資料) 【基準】'
            }
            
            name = strategy_names.get(strategy, strategy)
            print(f"{name:30s} {acc:6.4f}    {improvement:+.4f} ({improvement_pct:+.1f}%)")
        
        print("\n結論:")
        if results['zero_shot'] > baseline * 0.8:
            print("✓ 源模型在目標資料集上表現不錯，遷移學習有效！")
        else:
            print("⚠️  源模型直接遷移效果有限，需要微調")
        
        if results['fine_tune_10'] > results['zero_shot']:
            print("✓ 少量目標資料微調可以提升性能")
        
        if results['fine_tune_30'] > results['target_only']:
            print("✓ 遷移學習優於從頭訓練，證明源資料集的知識有幫助！")
        
        return results
    
    def save_transfer_model(self, model_name='transfer_model'):
        """儲存遷移學習模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_dict = {
            'source_model': self.source_model,
            'target_model': self.target_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_cols': self.feature_cols,
            'source_dataset': self.source_dataset_type,
            'target_dataset': self.target_dataset_type,
            'timestamp': timestamp
        }
        
        model_path = Path("models") / f"{model_name}_{timestamp}.pkl"
        joblib.dump(save_dict, model_path)
        
        print(f"\n💾 遷移模型已儲存: {model_path}")
        return model_path


def main():
    """主程式 - 互動式遷移學習"""
    
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║     Transfer Learning for Exoplanet Detection         ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    pipeline = TransferLearningPipeline()
    
    print("\n遷移學習實驗設定:")
    print("="*70)
    
    # 選擇源資料集
    print("\n源資料集（用於訓練）:")
    print("1. TESS (./tois.csv)")
    print("2. Kepler (./data/kepler_koi.csv)")
    print("3. 自訂路徑")
    
    source_choice = input("\n請選擇源資料集 (1-3): ").strip()
    
    if source_choice == '1':
        source_file = './tois.csv'
    elif source_choice == '2':
        source_file = './data/kepler_koi.csv'
    else:
        source_file = input("請輸入源資料集路徑: ").strip()
    
    # 選擇目標資料集
    print("\n目標資料集（用於測試/微調）:")
    print("1. TESS (./tois.csv)")
    print("2. Kepler (./data/kepler_koi.csv)")
    print("3. 自訂路徑")
    
    target_choice = input("\n請選擇目標資料集 (1-3): ").strip()
    
    if target_choice == '1':
        target_file = './tois.csv'
    elif target_choice == '2':
        target_file = './data/kepler_koi.csv'
    else:
        target_file = input("請輸入目標資料集路徑: ").strip()
    
    # 選擇標籤策略
    print("\n標籤策略:")
    print("1. 二分類 (PLANET vs NOT_PLANET)")
    print("2. 三分類 (PLANET, FALSE_POSITIVE, OTHER)")
    
    label_choice = input("\n請選擇 (1-2, 預設=1): ").strip()
    label_strategy = 'binary' if label_choice != '2' else 'three_class'
    
    # 選擇模型
    print("\n模型類型:")
    print("1. LightGBM (快速)")
    print("2. XGBoost (準確)")
    print("3. Random Forest (穩定)")
    
    model_choice = input("\n請選擇 (1-3, 預設=1): ").strip()
    model_map = {'1': 'lgbm', '2': 'xgb', '3': 'random_forest'}
    model_type = model_map.get(model_choice, 'lgbm')
    
    # 載入資料
    X_source, y_source, groups_source = pipeline.load_source_data(
        source_file, label_strategy=label_strategy
    )
    
    X_target, y_target, groups_target = pipeline.load_target_data(
        target_file, label_strategy=label_strategy
    )
    
    # 訓練源模型
    pipeline.train_source_model(X_source, y_source, model_type, n_estimators=200)
    
    # 選擇遷移策略
    print("\n遷移學習模式:")
    print("1. 零樣本測試（直接測試，不微調）")
    print("2. 微調模型（用部分目標資料微調）")
    print("3. 完整比較（測試所有策略）")
    
    transfer_choice = input("\n請選擇 (1-3, 預設=3): ").strip()
    
    if transfer_choice == '1':
        pipeline.test_on_target(X_target, y_target)
    
    elif transfer_choice == '2':
        ratio = input("\n微調資料比例 (0.1-0.5, 預設=0.2): ").strip()
        ratio = float(ratio) if ratio else 0.2
        pipeline.fine_tune_on_target(X_target, y_target, model_type, ratio)
    
    else:
        pipeline.compare_strategies(X_target, y_target, model_type)
    
    # 儲存
    save_choice = input("\n是否儲存遷移模型? (y/n): ").strip()
    if save_choice.lower() == 'y':
        pipeline.save_transfer_model()
    
    print("\n" + "="*70)
    print("✅ 遷移學習實驗完成！")
    print("="*70)


if __name__ == "__main__":
    main()