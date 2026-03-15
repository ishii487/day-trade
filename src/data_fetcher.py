import requests
import pandas as pd
from src.config import Config

class KabuDataFetcher:
    def __init__(self):
        # ↓もし.env経由で失敗し続けるなら、ここに直接 "あなたのパスワード" を書いてテストしてください
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
                raise Exception(f"トークン取得失敗(Status:{response.status_code}): {response.text}")
            return response.json()['Token']
        except requests.exceptions.ConnectionError:
            raise Exception(f"kabuステーションに接続できません。ポート {Config.API_PORT} を確認してください。")

    def save_and_merge_data(self, new_df, symbol):
        """最新データを既存のCSVに結合し、重複を排除して保存する"""
        file_path = os.path.join(Config.RAW_DATA_PATH, f"{symbol}_history.csv")
        
        if os.path.exists(file_path):
            # 既存のデータを読み込み
            old_df = pd.read_csv(file_path)
            # 結合
            combined_df = pd.concat([old_df, new_df], ignore_index=True)
            # 時刻等で重複を排除（'Time'カラムなど一意のキーがある場合）
            # もしTimeがない場合は 'High','Low','Close'などが完全に一致する行を削除
            combined_df = combined_df.drop_duplicates().reset_index(drop=True)
        else:
            combined_df = new_df
            
        combined_df.to_csv(file_path, index=False)
        print(f"データを更新しました。総件数: {len(combined_df)}件 (保存先: {file_path})")
        return combined_df

# src/data_fetcher.py (一部抜粋・修正)

    def fetch_historical_data(self, symbol, exchange):
        """指定した銘柄の分足データを取得する"""
        # ヒストリカルデータ取得エンドポイント
        # ※kabuステーションAPIの仕様にあわせ、?symbol=... の形式か
        # 銘柄登録済みのものから取得する設計にします
        url = f"{self.base_url}/historical/{symbol}@{exchange}"
        headers = {'X-API-KEY': self.token}
        
        params = {
            'period': '1'  # '1'は1分足。5分足なら'5'などを指定
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"データ取得エラー: {response.text}")
            return None
        
        data = response.json()
        
        # kabuステーションAPIのレスポンスからヒストリカルリスト(His)を抽出
        # APIのバージョンによって 'His' や 'Historical' などキー名が異なるため注意
        history = data.get('His', [])
        
        if not history:
            print("警告: ヒストリカルデータが空です。チャートを表示するか、銘柄登録を再確認してください。")
            return None

        # リストをDataFrameに変換
        df = pd.DataFrame(history)
        
        # APIのキー名を分析用(OHLCV)にマッピング
        # 例: 'T' (Time), 'O' (Open), 'H' (High), 'L' (Low), 'C' (Close), 'V' (Volume)
        column_mapping = {
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Open': 'Open',
            'Volume': 'Volume'
        }
        
        # 実際のAPIレスポンスのキー名に合わせて適宜リネーム
        # ここでは一例として、一般的なキー名を想定しています
        df = df.rename(columns={
            'HighPrice': 'High',
            'LowPrice': 'Low',
            'CurrentPrice': 'Close',
            'OpeningPrice': 'Open',
            'TradingVolume': 'Volume'
        })
        
        print(f"実データ {len(df)} 件を取得しました。")
        return df