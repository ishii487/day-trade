import requests
import pandas as pd
import os
from datetime import datetime
from src.config import Config

class KabuDataFetcher:
    def __init__(self):
        self.api_password = Config.API_PASSWORD
        self.base_url = Config.BASE_URL
        self.token = None

    def get_token(self):
        """APIトークンを取得する"""
        url = f"{self.base_url}/token"
        obj = {"APIPassword": self.api_password}
        try:
            response = requests.post(url, json=obj)
            if response.status_code == 200:
                self.token = response.json().get("Token")
                return self.token
            else:
                print(f"トークン取得エラー: {response.text}")
                return None
        except Exception as e:
            print(f"トークン取得接続エラー: {e}")
            return None

    def _register_symbol(self, symbol, exchange):
        """銘柄をAPI側に登録する（PUT /register）"""
        url = f"{self.base_url}/register"
        headers = {"X-API-KEY": self.token}
        obj = {"Symbols": [{"Symbol": symbol, "Exchange": exchange}]}
        requests.put(url, json=obj, headers=headers)

    def fetch_historical_data(self, symbol, exchange):
        """
        銘柄情報を取得し、学習用データ形式に変換する
        """
        if not self.get_token():
            return None

        # 1. 銘柄登録
        self._register_symbol(symbol, exchange)

        # 2. 銘柄詳細情報の取得（GET /board/{symbol}@{exchange}）
        # kabuステーションAPIで最も詳細な情報が得られる公式エンドポイント
        url = f"{self.base_url}/board/{symbol}@{exchange}"
        headers = {"X-API-KEY": self.token}

        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                
                # 時系列データとして扱うため、1行のDataFrameを作成
                # 学習には過去分が必要なため、このBotを実行し続けることでCSVに蓄積していきます
                new_row = {
                    'Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Open': data.get('CurrentPrice'),  # 当日始値などの項目があれば適宜修正
                    'High': data.get('HighPrice'),
                    'Low': data.get('LowPrice'),
                    'Close': data.get('CurrentPrice'),
                    'Volume': data.get('TradingVolume')
                }
                
                # 全ての値がNoneでないかチェック
                if new_row['Close'] is None:
                    print(f"[{symbol}] 市場が閉まっているか、データが取得できませんでした。")
                    return None
                    
                return pd.DataFrame([new_row])
            else:
                print(f"データ取得エラー: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"接続エラー: {e}")
            return None

    def save_and_merge_data(self, new_df, symbol):
        """既存のCSVと結合して保存する"""
        os.makedirs(Config.RAW_DATA_PATH, exist_ok=True)
        file_path = os.path.join(Config.RAW_DATA_PATH, f"{symbol}_history.csv")
        
        if os.path.exists(file_path):
            old_df = pd.read_csv(file_path)
            # 重複を排除（Dateが同じものは上書き）
            combined_df = pd.concat([old_df, new_df]).drop_duplicates(subset=['Date'], keep='last').reset_index(drop=True)
        else:
            combined_df = new_df
            
        combined_df.to_csv(file_path, index=False)
        return combined_df