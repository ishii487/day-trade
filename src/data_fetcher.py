import requests
import pandas as pd
from src.config import Config

class KabuDataFetcher:
    def __init__(self):
        self.pw = Config.API_PASSWORD 
        self.base_url = Config.BASE_URL
        self.token = self._get_token()

    def _get_token(self):
        """APIトークンを取得する"""
        url = f"{self.base_url}/token"
        if not self.pw:
            raise ValueError("APIパスワードが設定されていません。")
        
        obj = {'APIPassword': self.pw}
        try:
            response = requests.post(url, json=obj)
            if response.status_code != 200:
                raise Exception(f"トークン取得失敗: {response.text}")
            return response.json()['Token']
        except requests.exceptions.ConnectionError:
            raise Exception(f"kabuステーションに接続できません。ポート {Config.API_PORT} を確認してください。")

    def fetch_historical_data(self, symbol, exchange):
        """銘柄情報を取得し、分析用フォーマットに変換する"""
        url = f"{self.base_url}/board/{symbol}@{exchange}"
        headers = {'X-API-KEY': self.token}
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"データ取得エラー: {response.text}")
            return None
        
        data = response.json()
        
        # APIのキー名をプログラム用(OHLC)に変換
        df_row = {
            'High': data.get('HighPrice'),
            'Low': data.get('LowPrice'),
            'Close': data.get('CurrentPrice'),
            'Open': data.get('OpeningPrice'),
            'Volume': data.get('TradingVolume')
        }
        
        # DataFrame化
        df = pd.DataFrame([df_row])
        
        # 【重要】TA-Libの計算には最低14行以上のデータが必要なため、試運転用に水増し
        if len(df) < 20:
            df = pd.concat([df] * 20, ignore_index=True)
            
        return df