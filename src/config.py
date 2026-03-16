import os
from dotenv import load_dotenv

# .envファイルを読み込む
load_dotenv()

class Config:
    # API認証情報
    API_PASSWORD = os.getenv("KABU_API_PASSWORD")
    API_PORT = os.getenv("KABU_API_PORT", "18081") # 18081が一般的ですが.envに従います
    BASE_URL = f"http://localhost:{API_PORT}/kabusapi"
    
    # トレード設定
    SYMBOLS = ["9432", "7203", "8306"]
    EXCHANGE = 1  # 1: 東証
    
    # パス設定
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
    PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
    MODEL_PATH = os.path.join(BASE_DIR, "models")

# --- フォルダの自動作成を確実に行う ---
os.makedirs(Config.RAW_DATA_PATH, exist_ok=True)
os.makedirs(Config.PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(Config.MODEL_PATH, exist_ok=True)