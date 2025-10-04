"""
測試訓練好的系外行星檢測模型
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import json


class ExoplanetPredictor:
    """系外行星預測器"""
    
    def __init__(self, model_path):
        """載入模型"""
        print(f"📥 載入模型: {model_path}")
        
        model_data = joblib.load(model_path)
        
        # 處理不同的模型格式
        if 'model' in model_data:
            # 標準格式：單一模型
            self.model = model_data['model']
        elif 'target_model' in model_data:
            # Transfer learning 格式：使用目標模型
            self.model = model_data['target_model']
            if self.model is None:
                print("⚠️  目標模型為空，使用源模型")
                self.model = model_data['source_model']
        elif 'source_model' in model_data:
            # Transfer learning 格式：只有源模型
            self.model = model_data['source_model']
        else:
            raise ValueError("無法識別的模型格式")
        
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        
        # 特徵列表可能有不同的鍵名
        if 'selected_features' in model_data:
            self.selected_features = model_data['selected_features']
        elif 'feature_cols' in model_data:
            self.selected_features = model_data['feature_cols']
        else:
            raise ValueError("找不到特徵列表")
        
        self.version = model_data.get('version', 'unknown')
        self.data_source = model_data.get('data_source', 'unknown')
        
        print(f"✓ 模型載入成功")
        print(f"  版本: {self.version}")
        print(f"  資料來源: {self.data_source}")
        print(f"  特徵數: {len(self.selected_features)}")
        print(f"  類別: {list(self.label_encoder.classes_)}")
    
    def prepare_features(self, df):
        """準備特徵 - 支援 TESS 和 Kepler 資料集"""
        
        # TESS 和 Kepler 欄位對應
        feature_mapping = {
            # TESS 欄位
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
            'Stellar Distance (pc)': 'distance',
            # Kepler 欄位
            'koi_period': 'period',
            'koi_duration': 'duration',
            'koi_depth': 'depth',
            'koi_prad': 'planet_radius',
            'koi_teq': 'equil_temp',
            'koi_insol': 'insolation',
            'koi_model_snr': 'snr',
            'koi_steff': 'stellar_temp',
            'koi_srad': 'stellar_radius',
            'koi_slogg': 'stellar_logg',
            'koi_smass': 'stellar_mass',
            'koi_kepmag': 'tess_mag',  # 映射到相同特徵名
            'koi_dist': 'distance'
        }
        
        # 建立特徵矩陣
        available_features = {}
        for orig_name, new_name in feature_mapping.items():
            if orig_name in df.columns:
                if new_name not in available_features:  # 避免重複
                    available_features[orig_name] = new_name
        
        if not available_features:
            raise ValueError("找不到任何可用的特徵欄位")
        
        X = df[list(available_features.keys())].copy()
        X.columns = list(available_features.values())
        
        # 處理缺失值（用中位數填補）
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
        
        # 移除無限值
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # 補齊模型需要但缺失的特徵
        for feat in self.selected_features:
            if feat not in X.columns:
                X[feat] = 0  # 缺失特徵用0填充
        
        # 只選擇模型需要的特徵，並保持順序
        X_selected = X[self.selected_features]
        
        return X_selected
    
    def predict(self, X):
        """進行預測"""
        # 標準化
        X_scaled = self.scaler.transform(X)
        
        # 預測
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # 解碼標籤
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels, probabilities
    
    def predict_single(self, features_dict):
        """預測單筆資料"""
        df = pd.DataFrame([features_dict])
        X = self.prepare_features(df)
        labels, probs = self.predict(X)
        
        return labels[0], probs[0]


def test_on_file(model_path, test_file):
    """測試整個檔案"""
    print("\n" + "="*60)
    print("📊 批次測試")
    print("="*60)
    
    # 載入模型
    predictor = ExoplanetPredictor(model_path)
    
    # 載入測試資料
    print(f"\n📥 載入測試資料: {test_file}")
    df = pd.read_csv(test_file)
    print(f"✓ 測試資料: {df.shape}")
    
    # 準備特徵
    print("\n🔧 準備特徵...")
    X = predictor.prepare_features(df)
    print(f"✓ 特徵準備完成: {X.shape}")
    
    # 預測
    print("\n🎯 進行預測...")
    labels, probs = predictor.predict(X)
    
    # 加入結果到 DataFrame
    df['predicted_class'] = labels
    for i, class_name in enumerate(predictor.label_encoder.classes_):
        df[f'prob_{class_name}'] = probs[:, i]
    
    # 顯示結果摘要
    print("\n" + "="*60)
    print("📈 預測結果摘要")
    print("="*60)
    print(f"\n預測類別分佈:")
    print(df['predicted_class'].value_counts())
    
    # 如果有真實標籤，計算準確率
    if 'TESS Disposition' in df.columns:
        # 合併標籤（與訓練時相同）
        label_mapping = {
            'KP': 'PLANET', 'CP': 'PLANET', 'PC': 'PLANET',
            'EB': 'FALSE_POSITIVE', 'FP': 'FALSE_POSITIVE',
            'IS': 'OTHER', 'V': 'OTHER', 'O': 'OTHER'
        }
        df['true_class'] = df['TESS Disposition'].map(label_mapping)
        
        # 計算準確率
        correct = (df['predicted_class'] == df['true_class']).sum()
        total = len(df)
        accuracy = correct / total
        
        print(f"\n準確率: {accuracy:.2%} ({correct}/{total})")
        
        # 混淆矩陣
        from sklearn.metrics import confusion_matrix, classification_report
        print("\n分類報告:")
        print(classification_report(df['true_class'], df['predicted_class']))
    
    # 儲存結果
    output_file = "predictions_output.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 預測結果已儲存: {output_file}")
    
    # 顯示幾個範例
    print("\n" + "="*60)
    print("📋 預測範例 (前 10 筆)")
    print("="*60)
    
    display_cols = ['TOI', 'predicted_class'] + [f'prob_{c}' for c in predictor.label_encoder.classes_]
    if 'true_class' in df.columns:
        display_cols.insert(2, 'true_class')
    
    print(df[display_cols].head(10).to_string(index=False))
    
    return df


def test_single_example(model_path):
    """測試單筆範例"""
    print("\n" + "="*60)
    print("🔬 單筆測試")
    print("="*60)
    
    # 載入模型
    predictor = ExoplanetPredictor(model_path)
    
    # 範例資料（你可以修改這些值）
    example = {
        'Period (days)': 3.5,
        'Duration (hours)': 2.5,
        'Depth (ppm)': 5000,
        'Planet Radius (R_Earth)': 1.2,
        'Planet Equil Temp (K)': 1200,
        'Planet Insolation (Earth Flux)': 400,
        'Planet SNR': 25,
        'Stellar Eff Temp (K)': 5800,
        'Stellar Radius (R_Sun)': 1.1,
        'Stellar log(g) (cm/s^2)': 4.4,
        'Stellar Mass (M_Sun)': 1.0,
        'TESS Mag': 10.5,
        'Stellar Distance (pc)': 200
    }
    
    print("\n輸入特徵:")
    for key, value in example.items():
        print(f"  {key}: {value}")
    
    # 預測
    label, probs = predictor.predict_single(example)
    
    print(f"\n預測結果:")
    print(f"  類別: {label}")
    print(f"  機率分佈:")
    for i, class_name in enumerate(predictor.label_encoder.classes_):
        print(f"    {class_name}: {probs[i]:.2%}")
    
    return label, probs


def interactive_test(model_path):
    """互動式測試"""
    print("\n" + "="*60)
    print("💬 互動式測試")
    print("="*60)
    
    predictor = ExoplanetPredictor(model_path)
    
    print("\n請輸入系外行星候選的參數（按 Enter 使用預設值）:")
    
    features = {}
    defaults = {
        'Period (days)': 3.5,
        'Duration (hours)': 2.5,
        'Depth (ppm)': 5000,
        'Planet Radius (R_Earth)': 1.2,
        'Planet SNR': 25,
        'Stellar Eff Temp (K)': 5800,
        'Stellar Radius (R_Sun)': 1.1
    }
    
    for key, default in defaults.items():
        try:
            value = input(f"{key} (預設={default}): ").strip()
            features[key] = float(value) if value else default
        except:
            features[key] = default
    
    # 補充其他欄位（使用預設值）
    features.update({
        'Planet Equil Temp (K)': 1200,
        'Planet Insolation (Earth Flux)': 400,
        'Stellar log(g) (cm/s^2)': 4.4,
        'Stellar Mass (M_Sun)': 1.0,
        'TESS Mag': 10.5,
        'Stellar Distance (pc)': 200
    })
    
    label, probs = predictor.predict_single(features)
    
    print(f"\n🎯 預測結果: {label}")
    print(f"\n機率分佈:")
    for i, class_name in enumerate(predictor.label_encoder.classes_):
        bar = "█" * int(probs[i] * 50)
        print(f"  {class_name:20s} {probs[i]:6.2%} {bar}")


def main():
    """主程式"""
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║        Exoplanet Model Testing Tool                   ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    # 尋找所有模型
    model_dir = Path("models")
    model_files = list(model_dir.glob("*.pkl"))
    
    # 過濾掉非模型檔案
    model_keywords = ['exoplanet', 'combined', 'transfer', 'lgbm', 'xgb', 'stacking', 
                      'catboost', 'random_forest', 'model', 'new_model']
    model_files = [f for f in model_files if any(keyword in f.name.lower() for keyword in model_keywords)]
    
    if not model_files:
        print("❌ 找不到訓練好的模型！")
        print("請先執行訓練腳本:")
        print("  - python train_with_tess_data.py")
        print("  - python train_custom_model.py")
        print("  - python train_combined_model.py")
        return
    
    # 使用最新的模型
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    
    # 顯示所有可用模型
    if len(model_files) > 1:
        print("\n可用的模型:")
        for i, model_file in enumerate(sorted(model_files, key=lambda p: p.stat().st_mtime, reverse=True), 1):
            print(f"  {i}. {model_file.name}")
        
        print(f"\n預設使用最新模型: {latest_model.name}")
        use_default = input("使用預設模型? (y/n, 預設=y): ").strip().lower()
        
        if use_default == 'n':
            choice = input(f"選擇模型 (1-{len(model_files)}): ").strip()
            try:
                idx = int(choice) - 1
                sorted_models = sorted(model_files, key=lambda p: p.stat().st_mtime, reverse=True)
                latest_model = sorted_models[idx]
            except:
                print("無效選擇，使用預設模型")
    
    print(f"\n📦 使用模型: {latest_model}")
    
    print("\n請選擇測試模式:")
    print("1. 批次測試（測試整個 CSV 檔案）")
    print("2. 單筆測試（測試範例資料）")
    print("3. 互動式測試（手動輸入參數）")
    
    choice = input("\n請選擇 (1/2/3): ").strip()
    
    if choice == "1":
        test_file = input("請輸入測試檔案路徑 (預設=./tois.csv): ").strip()
        test_file = test_file if test_file else "./tois.csv"
        test_on_file(latest_model, test_file)
    
    elif choice == "2":
        test_single_example(latest_model)
    
    elif choice == "3":
        interactive_test(latest_model)
    
    else:
        print("無效的選擇！")
    
    print("\n" + "="*60)
    print("✅ 測試完成！")
    print("="*60)


if __name__ == "__main__":
    main()