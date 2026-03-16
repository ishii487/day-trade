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
    SYMBOLS = [
    "8306", "8316", "8411", "7182", "8354",  # banking
    "7203", "7267", "7201", "7270", "7261",  # auto
    "8035", "6857", "6146", "6723", "7735",  # semi
    "9432", "9433", "9434", "4443", "3994",  # telecom
    "9101", "9104", "9107", "9110", "9119"   # shipping
    ]
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