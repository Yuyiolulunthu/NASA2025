"""
生成範例系外行星資料
用於測試和展示（非真實資料）
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_sample_data(n_samples=5000, output_file='sample_exoplanet_data.csv'):
    """
    生成模擬的系外行星候選資料
    
    參數:
        n_samples: 資料筆數
        output_file: 輸出檔案名稱
    """
    
    print(f"🔧 生成 {n_samples} 筆範例資料...")
    
    np.random.seed(42)
    
    # 類別分佈（模擬真實分佈）
    # 假陽性通常最多，確認行星最少
    dispositions = np.random.choice(
        ['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED'],
        size=n_samples,
        p=[0.65, 0.25, 0.10]
    )
    
    # 為每個類別生成不同特性的資料
    data = []
    
    for i, disp in enumerate(dispositions):
        
        # 恆星 ID（每3-5個候選共享一個恆星）
        kepid = (i // 3) + 1000000
        
        if disp == 'CONFIRMED':
            # 確認行星：更明顯的信號
            period = np.random.lognormal(1.5, 1.2)  # 偏向較長週期
            depth = np.random.uniform(100, 3000)    # 較深的凌日
            snr = np.random.uniform(20, 150)        # 高信噪比
            radius = np.random.uniform(0.5, 15)     # 各種大小
            
        elif disp == 'CANDIDATE':
            # 候選：中等信號
            period = np.random.lognormal(1, 1.5)
            depth = np.random.uniform(50, 2000)
            snr = np.random.uniform(10, 80)
            radius = np.random.uniform(0.3, 20)
            
        else:  # FALSE POSITIVE
            # 假陽性：弱或異常信號
            period = np.random.lognormal(0.5, 2)    # 更隨機
            depth = np.random.uniform(20, 1500)     # 較淺
            snr = np.random.uniform(5, 50)          # 較低信噪比
            radius = np.random.uniform(0.1, 25)     # 更大範圍
        
        # 計算相關參數
        duration = period * np.random.uniform(0.02, 0.15)  # 凌日持續時間
        teq = np.random.uniform(200, 2500)                 # 平衡溫度
        insol = (5778 / teq) ** 4 * np.random.uniform(0.1, 100)  # 輻射
        
        # 恆星參數
        steff = np.random.normal(5500, 1000)  # 恆星溫度
        srad = np.random.lognormal(0, 0.3)    # 恆星半徑（太陽單位）
        
        # 加入一些雜訊和缺失值
        if np.random.random() < 0.1:  # 10% 缺失
            radius = np.nan
        if np.random.random() < 0.05:  # 5% 缺失
            teq = np.nan
        
        data.append({
            'kepid': kepid,
            'koi_period': period,
            'koi_duration': duration,
            'koi_depth': depth,
            'koi_prad': radius,
            'koi_teq': teq,
            'koi_insol': insol,
            'koi_model_snr': snr,
            'koi_steff': steff,
            'koi_srad': srad,
            'disposition': disp
        })
    
    # 建立 DataFrame
    df = pd.DataFrame(data)
    
    # 加入一些額外的資訊欄位
    df['koi_score'] = np.where(
        df['disposition'] == 'CONFIRMED', 
        np.random.uniform(0.8, 1.0, len(df)),
        np.where(
            df['disposition'] == 'CANDIDATE',
            np.random.uniform(0.5, 0.9, len(df)),
            np.random.uniform(0.0, 0.6, len(df))
        )
    )
    
    # 儲存
    df.to_csv(output_file, index=False)
    
    print(f"✅ 資料已儲存至: {output_file}")
    print(f"\n📊 資料統計:")
    print(f"   總筆數: {len(df)}")
    print(f"   類別分佈:")
    print(df['disposition'].value_counts())
    print(f"\n特徵範圍:")
    print(df.describe())
    
    return df


def generate_test_batch(n_samples=50, output_file='test_batch.csv'):
    """
    生成小批次測試資料（不含標籤）
    """
    print(f"\n🔧 生成 {n_samples} 筆測試批次...")
    
    np.random.seed(123)
    
    data = []
    for i in range(n_samples):
        data.append({
            'koi_period': np.random.lognormal(1, 1.5),
            'koi_duration': np.random.uniform(1, 10),
            'koi_depth': np.random.uniform(50, 2000),
            'koi_prad': np.random.uniform(0.5, 15),
            'koi_teq': np.random.uniform(300, 2000),
            'koi_insol': np.random.uniform(0.1, 50),
            'koi_model_snr': np.random.uniform(5, 100),
            'koi_steff': np.random.normal(5500, 1000),
            'koi_srad': np.random.lognormal(0, 0.3)
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    
    print(f"✅ 測試批次已儲存至: {output_file}")
    print(f"   筆數: {len(df)}")
    
    return df


def main():
    """主程式"""
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║     系外行星資料生成器                                  ║
    ║     Exoplanet Sample Data Generator                   ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    print("這將生成模擬的系外行星資料用於測試")
    print("注意：這不是真實的 NASA 資料\n")
    
    # 生成訓練資料
    df_train = generate_sample_data(
        n_samples=5000,
        output_file='sample_exoplanet_data.csv'
    )
    
    # 生成測試批次
    df_test = generate_test_batch(
        n_samples=50,
        output_file='test_batch.csv'
    )
    
    print("\n" + "="*60)
    print("✅ 完成！")
    print("="*60)
    print("\n下一步:")
    print("1. 使用 sample_exoplanet_data.csv 訓練模型")
    print("2. 使用 test_batch.csv 測試批次預測")
    print("\n如需真實 NASA 資料，請訪問:")
    print("https://exoplanetarchive.ipac.caltech.edu/")


if __name__ == "__main__":
    main()