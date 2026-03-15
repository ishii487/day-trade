import os
import time
import joblib
import pandas as pd
from datetime import datetime
from src.config import Config
from src.data_fetcher import KabuDataFetcher
from src.features import add_features

# 読み込むモデルのパス
MODEL_PATH = "models/latest.joblib"

def is_market_open():
    """現在の時刻が日本株の取引時間内か判定する"""
    now = datetime.now()
    # 平日のみ（月=0, ..., 金=4）
    if now.weekday() > 4:
        return False
        
    current_time = now.time()
    morning_session = datetime.strptime("09:00", "%H:%M").time() <= current_time <= datetime.strptime("11:30", "%H:%M").time()
    afternoon_session = datetime.strptime("12:30", "%H:%M").time() <= current_time <= datetime.strptime("15:00", "%H:%M").time()
    
    return morning_session or afternoon_session

def run_bot():
    print("=== AutoDayTrade Bot 起動 ===")
    
    if not os.path.exists(MODEL_PATH):
        print(f"エラー: モデルが見つかりません。先に学習(main.py)を実行して {MODEL_PATH} を作成してください。")
        return

    # モデルと特徴量リストの読み込み
    print(f"モデルをロード中: {MODEL_PATH}")
    saved_data = joblib.load(MODEL_PATH)
    model = saved_data['model']
    required_features = saved_data['features']
    print(f"使用する特徴量: {required_features}")

    fetcher = KabuDataFetcher()
    symbol = Config.SYMBOL

    print("監視を開始します。終了する場合は Ctrl+C を押してください。\n")

    try:
        while True:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 市場時間外はスリープして待機
            if not is_market_open():
                print(f"[{now_str}] 市場時間外です。待機中...")
                time.sleep(60) # 1分待機
                continue

            # 1. 最新データの取得
            print(f"[{now_str}] データを取得中...")
            df_live = fetcher.fetch_historical_data(symbol, Config.EXCHANGE)
            
            if df_live is None or df_live.empty:
                print("データの取得に失敗しました。リトライします。")
                time.sleep(10)
                continue

            # 2. 特徴量の計算
            df_live = add_features(df_live)
            
            # データ不足時のガード（※TA-Libの計算に過去の行数が必要なため）
            if len(df_live) < 50:
                print("警告: 特徴量計算に必要な行数が不足しています。過去のデータを結合してください。")
                time.sleep(60)
                continue
                
            # 最新の1行（直近の足）を取得
            latest_row = df_live.iloc[-1:]
            
            # 3. 予測の実行
            # 学習時と全く同じ特徴量のみをモデルに渡す
            X_live = latest_row[required_features]
            pred_prob = model.predict(X_live)[0]
            
            # 4. シグナルの判定と出力
            current_price = latest_row['Close'].values[0]
            
            if pred_prob > 0.5:
                print(f"📈 【BUY SIGNAL】 現在値: {current_price}円 | 上昇確率: {pred_prob*100:.1f}%")
                # --------------------------------------------------
                # ここにkabuステーションAPIの「買い注文」を入れる処理を追加します
                # --------------------------------------------------
            else:
                print(f"⏳ 待機 (確率: {pred_prob*100:.1f}%) | 現在値: {current_price}円")

            # 1分（または任意のインターバル）待機
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\n=== Botを停止しました ===")

if __name__ == "__main__":
    run_bot()