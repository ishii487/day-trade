import yfinance as yf
import os
import time
from src.config import Config

# --- 業界別・銘柄構成の定義 ---
# ここに銘柄を追加するだけで、自動的に業界フォルダが作られ、データが蓄積されます
SECTORS = {
    "banking": ["8306", "8316", "8411", "7182", "8354"],
    "auto": ["7203", "7267", "7201", "7270", "7261"],
    "telecom": ["9432", "9433", "9434", "4443", "3994"],
    "semi": ["8035", "6857", "6146", "6723", "7735"], # スクリーン(7735)を追加
    "shipping": ["9101", "9104", "9107", "9110", "9119"] # 飯野海運(9119)を追加
}

def import_sector_data():
    """
    SECTORSで定義した全銘柄の1分足データを、業界別フォルダに保存する
    """
    base_raw_path = Config.RAW_DATA_PATH # data/raw/
    
    for sector_name, symbols in SECTORS.items():
        print(f"\n--- 業界: {sector_name} のデータ取得開始 ---")
        
        # 業界ごとの保存フォルダを作成 (例: data/raw/banking/)
        sector_path = os.path.join(base_raw_path, sector_name)
        os.makedirs(sector_path, exist_ok=True)
        
        for symbol in symbols:
            print(f"  [{symbol}] Yahoo Financeから1分足(直近7日分)を取得中...")
            
            try:
                # interval="1m" は直近7日間まで取得可能
                df = yf.download(f"{symbol}.T", interval="1m", period="7d")
                
                if df.empty:
                    print(f"  [{symbol}] データが空です。スキップします。")
                    continue
                
                # マルチインデックスの解消とカラム整え
                df = df.reset_index()
                df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                
                # 日時フォーマットの統一
                df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # 保存 (例: data/raw/banking/8306_history.csv)
                file_path = os.path.join(sector_path, f"{symbol}_history.csv")
                df.to_csv(file_path, index=False)
                print(f"  [{symbol}] 保存完了: {len(df)}行 -> {file_path}")
                
                # API負荷軽減のための待機
                time.sleep(1)
                
            except Exception as e:
                print(f"  [{symbol}] エラーが発生しました: {e}")

    print("\n=== 全セクターのデータ収集が完了しました ===")

if __name__ == "__main__":
    import_sector_data()