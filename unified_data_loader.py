"""
統一的資料載入器
自動處理不同資料集的格式差異
"""

import pandas as pd
import numpy as np
from pathlib import Path


class UnifiedDataLoader:
    """統一資料載入器 - 支援多種系外行星資料集"""
    
    def __init__(self):
        self.dataset_configs = {
            'tess': {
                'label_column': 'TESS Disposition',
                'features': {
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
                },
                'group_column': 'TIC ID'
            },
            'kepler': {
                'label_column': 'koi_disposition',
                'features': {
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
                    'koi_kepmag': 'kepler_mag',
                    'koi_dist': 'distance'
                },
                'group_column': 'kepid'
            },
            'k2': {
                'label_column': 'k2c_disp',
                'features': {
                    'pl_orbper': 'period',
                    'pl_trandur': 'duration',
                    'pl_trandep': 'depth',
                    'pl_rade': 'planet_radius',
                    'pl_eqt': 'equil_temp',
                    'pl_insol': 'insolation',
                    'st_teff': 'stellar_temp',
                    'st_rad': 'stellar_radius',
                    'st_logg': 'stellar_logg',
                    'st_mass': 'stellar_mass',
                    'sy_kmag': 'k2_mag',
                    'sy_dist': 'distance'
                },
                'group_column': 'epic_hostname'
            },
            'confirmed': {
                'label_column': None,  # 所有都是已確認的行星
                'features': {
                    'pl_orbper': 'period',
                    'pl_trandur': 'duration',
                    'pl_trandep': 'depth',
                    'pl_rade': 'planet_radius',
                    'pl_eqt': 'equil_temp',
                    'pl_insol': 'insolation',
                    'st_teff': 'stellar_temp',
                    'st_rad': 'stellar_radius',
                    'st_logg': 'stellar_logg',
                    'st_mass': 'stellar_mass',
                    'sy_vmag': 'v_mag',
                    'sy_dist': 'distance'
                },
                'group_column': 'hostname'
            }
        }
    
    def detect_dataset_type(self, df):
        """自動偵測資料集類型"""
        columns = set(df.columns)
        
        if 'TESS Disposition' in columns or 'TOI' in columns:
            return 'tess'
        elif 'koi_disposition' in columns or 'kepid' in columns:
            return 'kepler'
        elif 'k2c_disp' in columns or 'epic_hostname' in columns:
            return 'k2'
        elif 'pl_name' in columns and 'disc_facility' in columns:
            return 'confirmed'
        else:
            return None
    
    def load_dataset(self, filepath, dataset_type=None, label_strategy='three_class'):
        """
        載入資料集
        
        Args:
            filepath: 資料檔案路徑
            dataset_type: 資料集類型 ('tess', 'kepler', 'k2', 'confirmed')
                         如果為 None，會自動偵測
            label_strategy: 標籤策略
                - 'binary': 二分類
                - 'three_class': 三分類
                - 'full': 保留原始類別
        
        Returns:
            處理好的 DataFrame
        """
        print(f"\n📥 載入資料: {filepath}")
        
        # 讀取資料
        df = pd.read_csv(filepath)
        print(f"✓ 原始資料: {df.shape}")
        
        # 自動偵測資料集類型
        if dataset_type is None:
            dataset_type = self.detect_dataset_type(df)
            print(f"✓ 偵測到資料集類型: {dataset_type}")
        
        if dataset_type not in self.dataset_configs:
            raise ValueError(f"不支援的資料集類型: {dataset_type}")
        
        config = self.dataset_configs[dataset_type]
        
        # 處理標籤
        df = self._process_labels(df, dataset_type, config, label_strategy)
        
        # 特徵工程
        df = self._engineer_features(df, config)
        
        print(f"✓ 處理完成: {df.shape}")
        print(f"\n類別分佈:")
        if 'disposition' in df.columns:
            print(df['disposition'].value_counts())
        
        return df, dataset_type
    
    def _process_labels(self, df, dataset_type, config, label_strategy):
        """處理標籤欄位"""
        
        if dataset_type == 'confirmed':
            # 所有確認的行星都標記為 PLANET
            df['disposition'] = 'PLANET'
            return df
        
        label_col = config['label_column']
        
        if label_col not in df.columns:
            raise ValueError(f"找不到標籤欄位: {label_col}")
        
        # 移除空標籤
        df = df.dropna(subset=[label_col])
        
        # 根據資料集類型和策略處理標籤
        if dataset_type == 'tess':
            df = self._process_tess_labels(df, label_col, label_strategy)
        elif dataset_type == 'kepler':
            df = self._process_kepler_labels(df, label_col, label_strategy)
        elif dataset_type == 'k2':
            df = self._process_k2_labels(df, label_col, label_strategy)
        
        return df
    
    def _process_tess_labels(self, df, label_col, strategy):
        """處理 TESS 標籤"""
        if strategy == 'binary':
            mapping = {
                'KP': 'PLANET', 'CP': 'PLANET', 'PC': 'PLANET',
                'EB': 'NOT_PLANET', 'FP': 'NOT_PLANET',
                'IS': 'NOT_PLANET', 'V': 'NOT_PLANET', 'O': 'NOT_PLANET'
            }
        elif strategy == 'three_class':
            mapping = {
                'KP': 'PLANET', 'CP': 'PLANET', 'PC': 'PLANET',
                'EB': 'FALSE_POSITIVE', 'FP': 'FALSE_POSITIVE',
                'IS': 'OTHER', 'V': 'OTHER', 'O': 'OTHER'
            }
        else:  # full
            mapping = None
        
        if mapping:
            df['disposition'] = df[label_col].map(mapping)
            df = df.dropna(subset=['disposition'])
        else:
            df['disposition'] = df[label_col]
        
        return df
    
    def _process_kepler_labels(self, df, label_col, strategy):
        """處理 Kepler 標籤"""
        if strategy == 'binary':
            mapping = {
                'CONFIRMED': 'PLANET',
                'CANDIDATE': 'PLANET',
                'FALSE POSITIVE': 'NOT_PLANET'
            }
        elif strategy == 'three_class':
            mapping = {
                'CONFIRMED': 'PLANET',
                'CANDIDATE': 'CANDIDATE',
                'FALSE POSITIVE': 'FALSE_POSITIVE'
            }
        else:  # full
            mapping = None
        
        if mapping:
            df['disposition'] = df[label_col].map(mapping)
            df = df.dropna(subset=['disposition'])
        else:
            df['disposition'] = df[label_col]
        
        return df
    
    def _process_k2_labels(self, df, label_col, strategy):
        """處理 K2 標籤"""
        # K2 的標籤格式類似 Kepler
        return self._process_kepler_labels(df, label_col, strategy)
    
    def _engineer_features(self, df, config):
        """特徵工程"""
        feature_mapping = config['features']
        
        # 選擇存在的特徵
        available_features = {}
        for orig_name, new_name in feature_mapping.items():
            if orig_name in df.columns:
                available_features[orig_name] = new_name
        
        if not available_features:
            print("⚠️  警告: 沒有找到可用的特徵欄位")
            return df
        
        # 建立特徵矩陣
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
        
        # 移除無限值
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # 將特徵加回原始 DataFrame
        for col in X.columns:
            df[col] = X[col]
        
        return df
    
    def get_features_and_labels(self, df, dataset_type):
        """提取特徵和標籤"""
        config = self.dataset_configs[dataset_type]
        
        # 獲取所有特徵欄位（原始特徵 + 衍生特徵）
        feature_cols = list(config['features'].values())
        
        # 添加衍生特徵
        derived_features = [
            'duration_period_ratio', 'depth_radius_ratio', 
            'log_snr', 'temp_insol_ratio', 'stellar_luminosity'
        ]
        
        feature_cols.extend([f for f in derived_features if f in df.columns])
        
        # 只保留存在的欄位
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[feature_cols]
        y = df['disposition'] if 'disposition' in df.columns else None
        
        # 獲取分組欄位
        group_col = config['group_column']
        groups = df[group_col] if group_col in df.columns else None
        
        return X, y, groups


def quick_load(filepath, dataset_type=None, label_strategy='three_class'):
    """快速載入資料的便捷函數"""
    loader = UnifiedDataLoader()
    df, detected_type = loader.load_dataset(filepath, dataset_type, label_strategy)
    X, y, groups = loader.get_features_and_labels(df, detected_type)
    
    return X, y, groups, detected_type


def main():
    """示範使用方法"""
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║        Unified Data Loader Demo                       ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    print("\n支援的資料集:")
    print("1. TESS TOI (tess)")
    print("2. Kepler KOI (kepler)")
    print("3. K2 Candidates (k2)")
    print("4. Confirmed Planets (confirmed)")
    
    print("\n範例使用:")
    print("""
# 自動偵測資料集類型
from unified_data_loader import quick_load

X, y, groups, dataset_type = quick_load('./tois.csv', label_strategy='binary')
print(f"載入 {dataset_type} 資料集")
print(f"特徵形狀: {X.shape}")
print(f"標籤分佈:\\n{y.value_counts()}")

# 或者明確指定資料集類型
X, y, groups, _ = quick_load('./kepler.csv', dataset_type='kepler', label_strategy='three_class')
    """)
    
    # 實際測試
    test_file = input("\n請輸入資料檔案路徑 (或按 Enter 跳過): ").strip()
    
    if test_file and Path(test_file).exists():
        try:
            X, y, groups, dataset_type = quick_load(test_file)
            print(f"\n✓ 成功載入 {dataset_type} 資料集")
            print(f"✓ 特徵形狀: {X.shape}")
            if y is not None:
                print(f"✓ 標籤分佈:\n{y.value_counts()}")
        except Exception as e:
            print(f"\n❌ 載入失敗: {e}")


if __name__ == "__main__":
    main()