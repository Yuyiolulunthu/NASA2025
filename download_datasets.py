"""
系外行星資料集下載器
支援多個 NASA 資料集和 Kaggle 資料集
"""

import requests
import pandas as pd
from pathlib import Path
import json
from datetime import datetime


class ExoplanetDatasetDownloader:
    """系外行星資料集下載器"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # NASA Exoplanet Archive API
        self.nasa_base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        
        # 可用的資料集配置
        self.datasets = {
            '1_tess_toi': {
                'name': 'TESS Objects of Interest (TOI)',
                'description': 'TESS 望遠鏡發現的系外行星候選',
                'table': 'toi',
                'size': '~7,700 筆',
                'url': 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI',
                'filename': 'tess_toi.csv'
            },
            '2_kepler_koi': {
                'name': 'Kepler Objects of Interest (KOI)',
                'description': 'Kepler 望遠鏡發現的系外行星候選',
                'table': 'cumulative',
                'size': '~10,000 筆',
                'url': 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative',
                'filename': 'kepler_koi.csv'
            },
            '3_k2_candidates': {
                'name': 'K2 Candidates',
                'description': 'K2 任務發現的系外行星候選',
                'table': 'k2candidates',
                'size': '~1,000 筆',
                'url': 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2candidates',
                'filename': 'k2_candidates.csv'
            },
            '4_confirmed_planets': {
                'name': 'Confirmed Exoplanets',
                'description': '所有已確認的系外行星（綜合資料）',
                'table': 'pscomppars',
                'size': '~5,600+ 筆',
                'url': 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PS',
                'filename': 'confirmed_planets.csv'
            },
            '5_extended_catalog': {
                'name': 'Extended Exoplanet Catalog',
                'description': '擴展的系外行星目錄',
                'table': 'exoplanets',
                'size': '~5,000+ 筆',
                'url': 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=exoplanets',
                'filename': 'extended_catalog.csv'
            },
            '6_microlensing': {
                'name': 'Microlensing Planets',
                'description': '透過微重力透鏡發現的系外行星',
                'table': 'microlensing',
                'size': '~200 筆',
                'url': 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=ML',
                'filename': 'microlensing.csv'
            }
        }
    
    def list_datasets(self):
        """列出所有可用的資料集"""
        print("\n" + "="*70)
        print("📊 可用的系外行星資料集")
        print("="*70)
        
        for key, info in self.datasets.items():
            print(f"\n{key}. {info['name']}")
            print(f"   描述: {info['description']}")
            print(f"   大小: {info['size']}")
            print(f"   檔案: {info['filename']}")
    
    def download_from_nasa(self, table_name, filename):
        """從 NASA Exoplanet Archive 下載資料"""
        print(f"\n📥 下載 {table_name} 資料...")
        
        # 使用 CSV 格式下載
        query = f"SELECT * FROM {table_name}"
        
        params = {
            'query': query,
            'format': 'csv'
        }
        
        try:
            response = requests.get(self.nasa_base_url, params=params, timeout=60)
            response.raise_for_status()
            
            filepath = self.data_dir / filename
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # 驗證下載
            df = pd.read_csv(filepath)
            print(f"✓ 下載成功: {filepath}")
            print(f"✓ 資料形狀: {df.shape}")
            print(f"✓ 欄位數: {len(df.columns)}")
            
            return filepath
            
        except Exception as e:
            print(f"❌ 下載失敗: {e}")
            print("\n💡 備用方案: 使用網頁下載")
            print(f"請訪問: {self.datasets[list(self.datasets.keys())[0]]['url']}")
            return None
    
    def download_dataset(self, dataset_key):
        """下載指定的資料集"""
        if dataset_key not in self.datasets:
            print(f"❌ 未知的資料集: {dataset_key}")
            return None
        
        info = self.datasets[dataset_key]
        print(f"\n{'='*70}")
        print(f"正在下載: {info['name']}")
        print(f"{'='*70}")
        
        filepath = self.download_from_nasa(info['table'], info['filename'])
        
        if filepath:
            # 保存元資料
            metadata = {
                'dataset_name': info['name'],
                'description': info['description'],
                'download_date': datetime.now().isoformat(),
                'filepath': str(filepath),
                'source': 'NASA Exoplanet Archive'
            }
            
            metadata_file = self.data_dir / f"{info['filename']}.meta.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return filepath
    
    def download_kaggle_dataset(self):
        """下載 Kaggle 系外行星資料集"""
        print("\n" + "="*70)
        print("📦 Kaggle 系外行星資料集")
        print("="*70)
        
        print("\n可用的 Kaggle 資料集:")
        print("\n1. Kepler Exoplanet Search Results")
        print("   https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results")
        print("   - 完整的 Kepler 資料集")
        print("   - 包含特徵工程後的資料")
        
        print("\n2. TESS Exoplanet Candidates")
        print("   https://www.kaggle.com/datasets/quadeer15sh/tess-exoplanet-candidates")
        print("   - TESS 資料集")
        
        print("\n3. Exoplanet Hunting in Deep Space")
        print("   https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data")
        print("   - 時序資料（光變曲線）")
        
        print("\n💡 如何下載 Kaggle 資料集:")
        print("1. 安裝 Kaggle CLI: pip install kaggle")
        print("2. 設定 API token: https://www.kaggle.com/docs/api")
        print("3. 下載: kaggle datasets download -d nasa/kepler-exoplanet-search-results")
    
    def download_all(self):
        """下載所有 NASA 資料集"""
        print("\n開始下載所有資料集...")
        
        results = {}
        for key in self.datasets.keys():
            filepath = self.download_dataset(key)
            results[key] = filepath
        
        print("\n" + "="*70)
        print("📊 下載摘要")
        print("="*70)
        
        for key, filepath in results.items():
            status = "✓" if filepath else "✗"
            print(f"{status} {self.datasets[key]['name']}: {filepath}")
        
        return results
    
    def create_combined_dataset(self):
        """建立綜合資料集（合併多個來源）"""
        print("\n" + "="*70)
        print("🔗 建立綜合資料集")
        print("="*70)
        
        # 這裡可以實作將多個資料集合併的邏輯
        # 例如：合併 TESS, Kepler, K2 的資料
        
        print("\n⚠️  此功能尚未實作")
        print("💡 提示: 不同資料集的欄位名稱可能不同，需要仔細對齊")


def download_specific_datasets():
    """下載特定資料集的便捷方法"""
    
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║        Exoplanet Dataset Downloader                   ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    downloader = ExoplanetDatasetDownloader()
    
    # 顯示可用資料集
    downloader.list_datasets()
    
    # Kaggle 資料集
    print("\n" + "="*70)
    downloader.download_kaggle_dataset()
    
    print("\n" + "="*70)
    print("請選擇操作:")
    print("="*70)
    print("1. 下載單一資料集")
    print("2. 下載所有 NASA 資料集")
    print("3. 查看 Kaggle 資料集資訊")
    print("0. 退出")
    
    choice = input("\n請選擇 (0-3): ").strip()
    
    if choice == '1':
        dataset_key = input("\n請輸入資料集編號 (例如: 1_tess_toi): ").strip()
        downloader.download_dataset(dataset_key)
    
    elif choice == '2':
        confirm = input("\n⚠️  這將下載所有資料集，可能需要幾分鐘。繼續? (y/n): ").strip()
        if confirm.lower() == 'y':
            downloader.download_all()
    
    elif choice == '3':
        downloader.download_kaggle_dataset()
    
    print("\n✅ 完成！")
    print(f"\n資料已儲存在: {downloader.data_dir}")


if __name__ == "__main__":
    download_specific_datasets()