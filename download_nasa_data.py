"""
從 NASA Exoplanet Archive 下載真實資料 - 修復版
"""

import requests
import pandas as pd
from pathlib import Path
import time

def download_kepler_simple():
    """使用簡化的直接 URL 下載 Kepler 資料"""
    
    print("🛰️  從 NASA Exoplanet Archive 下載 Kepler 資料...")
    print("="*60)
    
    # 使用直接的 CSV 下載 URL
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv"
    
    try:
        print("📥 正在下載... (這可能需要 1-2 分鐘)")
        
        response = requests.get(url, timeout=180)
        response.raise_for_status()
        
        # 儲存到檔案
        output_file = "kepler_data.csv"
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        # 讀取並顯示資訊
        df = pd.read_csv(output_file)
        
        # 只保留有 disposition 的資料
        df = df[df['koi_disposition'].notna()]
        df.to_csv(output_file, index=False)
        
        print(f"\n✅ 下載成功！")
        print(f"📁 檔案: {output_file}")
        print(f"📊 資料筆數: {len(df):,}")
        print(f"\n類別分佈:")
        print(df['koi_disposition'].value_counts())
        
        print(f"\n重要欄位:")
        important_cols = ['kepid', 'koi_disposition', 'koi_period', 'koi_duration', 
                         'koi_depth', 'koi_prad', 'koi_teq', 'koi_model_snr']
        for col in important_cols:
            if col in df.columns:
                missing = df[col].isna().sum()
                print(f"  ✓ {col} (缺失: {missing})")
        
        return df
        
    except requests.exceptions.Timeout:
        print("❌ 下載超時")
        return None
    except Exception as e:
        print(f"❌ 下載失敗: {e}")
        return None


def download_via_astroquery():
    """使用 astroquery 套件下載（備用方案）"""
    
    print("\n🛰️  使用 astroquery 下載...")
    print("="*60)
    
    try:
        from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
        
        print("📥 正在查詢 Kepler 累積目錄...")
        
        # 查詢所有 Kepler 目標
        kepler = NasaExoplanetArchive.query_criteria(
            table="cumulative",
            select="*"
        )
        
        # 轉換為 DataFrame
        df = kepler.to_pandas()
        
        # 只保留有 disposition 的
        df = df[df['koi_disposition'].notna()]
        
        # 儲存
        output_file = "kepler_data.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\n✅ 下載成功！")
        print(f"📁 檔案: {output_file}")
        print(f"📊 資料筆數: {len(df):,}")
        print(f"\n類別分佈:")
        print(df['koi_disposition'].value_counts())
        
        return df
        
    except ImportError:
        print("❌ 需要安裝 astroquery")
        print("執行: pip install astroquery")
        return None
    except Exception as e:
        print(f"❌ 下載失敗: {e}")
        return None


def show_manual_instructions():
    """顯示手動下載說明"""
    print("\n" + "="*60)
    print("📖 手動下載說明")
    print("="*60)
    print("""
如果自動下載失敗，請手動下載：

方法 1: 使用 Web 介面
─────────────────────
1. 訪問: https://exoplanetarchive.ipac.caltech.edu/
2. 點擊 "Data" → "Kepler Objects of Interest"
3. 或直接訪問: 
   https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative
4. 點擊右上角 "Download Table"
5. 選擇 "CSV Format"
6. 下載後重命名為 kepler_data.csv
7. 放到當前目錄: D:\\2025_NASA\\V2\\

方法 2: 使用直接連結
─────────────────────
複製以下網址到瀏覽器：

https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+kepid,kepoi_name,koi_disposition,koi_period,koi_time0bk,koi_duration,koi_depth,koi_prad,koi_teq,koi_insol,koi_model_snr,koi_steff,koi_slogg,koi_srad,koi_score+from+cumulative+where+koi_disposition+is+not+null&format=csv

這會直接下載 CSV 檔案。

方法 3: 使用 wget 或 curl (如果已安裝)
─────────────────────────────────────
curl "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv" -o kepler_data.csv

下載完成後，確認檔案：
- 檔名: kepler_data.csv
- 位置: D:\\2025_NASA\\V2\\
- 大小: 約 5-10 MB
- 包含約 9,000+ 筆資料
""")


def verify_data(filepath='kepler_data.csv'):
    """驗證下載的資料"""
    print("\n🔍 驗證資料...")
    
    if not Path(filepath).exists():
        print(f"❌ 找不到 {filepath}")
        return False
    
    try:
        df = pd.read_csv(filepath, nrows=5)
        
        required_cols = ['koi_disposition', 'koi_period', 'koi_depth']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️  缺少必要欄位: {missing_cols}")
            return False
        
        print(f"✅ 資料格式正確")
        print(f"✅ 包含必要欄位")
        
        # 讀取完整資料
        df_full = pd.read_csv(filepath)
        print(f"✅ 共 {len(df_full):,} 筆資料")
        
        return True
        
    except Exception as e:
        print(f"❌ 資料驗證失敗: {e}")
        return False


def main():
    """主程式"""
    
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║     NASA 系外行星資料下載器                            ║
    ║     NASA Exoplanet Data Downloader                    ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    # 檢查是否已有資料
    if Path('kepler_data.csv').exists():
        print("✓ 發現已存在的 kepler_data.csv")
        if verify_data():
            print("\n✅ 資料已就緒，可以開始訓練！")
            print("\n下一步: python train_with_nasa_data.py")
            return
        else:
            print("\n⚠️  現有資料可能有問題，重新下載...")
    
    print("\n選擇下載方式：")
    print("  1 - 自動下載 (簡化版)")
    print("  2 - 使用 astroquery (需先安裝)")
    print("  3 - 顯示手動下載說明")
    
    choice = input("\n請選擇 (1/2/3): ").strip()
    
    success = False
    
    if choice == '1':
        df = download_kepler_simple()
        success = df is not None
    elif choice == '2':
        df = download_via_astroquery()
        success = df is not None
    elif choice == '3':
        show_manual_instructions()
        print("\n手動下載後，執行此腳本驗證資料:")
        print("python download_nasa_data.py")
        return
    else:
        print("無效選擇")
        return
    
    if success:
        print("\n" + "="*60)
        print("✅ 完成！")
        print("="*60)
        print("\n下一步:")
        print("python train_with_nasa_data.py")
    else:
        print("\n" + "="*60)
        print("⚠️  自動下載失敗")
        print("="*60)
        show_manual_instructions()


if __name__ == "__main__":
    main()