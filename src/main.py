# src/main.py
from src.config import Config
from src.data_fetcher import KabuDataFetcher
from src.features import add_features, add_dynamic_target
from src.train import objective
import optuna
import pandas as pd
import sys

def main():
    print("--- 処理を開始します ---")
    
    # 1. データ取得
    fetcher = KabuDataFetcher()
    print(f"銘柄 {Config.SYMBOL} のデータを取得中...")
    raw_data = fetcher.fetch_historical_data(Config.SYMBOL, Config.EXCHANGE)
    
    if raw_data is None:
        print("エラー: データの取得に失敗しました。kabuステーションが起動しているか確認してください。")
        return

    # APIのレスポンス形式を確認するために表示
    print(f"取得データ件数: {len(raw_data)}")
    
    # 2. DataFrame化
    # ※APIの仕様に基づき、レスポンスのリストが入っているキーを指定する必要があります
    # 仮にレスポンスがリストそのものであれば以下でOK
    try:
        df = pd.DataFrame(raw_data)
        if df.empty:
            print("エラー: データが空です。銘柄登録が済んでいるか確認してください。")
            return
    except Exception as e:
        print(f"DataFrame変換エラー: {e}")
        return
    
    print("特徴量を付与中...")
    df = add_features(df)
    
    # 3. 最適化
    print("Optunaによる最適化を開始します (試行回数: 50)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, df), n_trials=50)
    
    print("--- 最適化完了 ---")
    print(f"最良スコア: {study.best_value}")
    print(f"最良パラメータ: {study.best_params}")

if __name__ == "__main__":
    main()