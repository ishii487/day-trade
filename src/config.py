import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API認証情報
    API_PASSWORD = os.getenv("KABU_API_PASSWORD")
    API_PORT = os.getenv("KABU_API_PORT", "18080")
    BASE_URL = f"http://localhost:{API_PORT}/kabusapi"
    
    # トレード設定
    SYMBOL = "9432"  # 例: NTT（ターゲット銘柄）
    EXCHANGE = 1     # 1: 東証
    
    # データ保存パス
    RAW_DATA_PATH = "data/raw/"
    PROCESSED_DATA_PATH = "data/processed/"

# フォルダが存在しない場合は作成
os.makedirs(Config.RAW_DATA_PATH, exist_ok=True)
os.makedirs(Config.PROCESSED_DATA_PATH, exist_ok=True)