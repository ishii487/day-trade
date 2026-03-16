import os
import time
import joblib
import pandas as pd
import optuna
import lightgbm as lgb
from datetime import datetime
from src.config import Config
from src.data_fetcher import KabuDataFetcher
from src.features import add_features
from src.train import objective, get_dynamic_features_ranked
from src.utils.notifier import send_line_notification

def main():
    print("=== AutoDayTrade マルチシンボル学習システム起動 ===")
    fetcher = KabuDataFetcher()
    os.makedirs(Config.MODEL_PATH, exist_ok=True)
    os.makedirs(Config.RAW_DATA_PATH, exist_ok=True)
    
    # 期待されるカラム名のリストと、Excel用日本語変換マップ
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    column_map = {
        '日付': 'Date', '日時': 'Date', '時刻': 'Date',
        '始値': 'Open', '寄付': 'Open',
        '高値': 'High',
        '安値': 'Low',
        '終値': 'Close', '現在値': 'Close',
        '出来高': 'Volume', '売買高': 'Volume', 'TradingVolume': 'Volume'
    }

    for symbol in Config.SYMBOLS:
        print(f"\n--- 銘柄 [{symbol}] の学習プロセスを開始 ---")
        
        # 1. APIからのデータ取得（日中の運用時用）
        new_data_df = fetcher.fetch_historical_data(symbol, Config.EXCHANGE)
        
        file_path = os.path.join(Config.RAW_DATA_PATH, f"{symbol}_history.csv")
        
        # 2. 既存のCSVデータの読み込み（Excelアドイン等のデータ）
        if os.path.exists(file_path):
            try:
                # 日本語のCSV（Shift-JIS）に対応させるため encoding='cp932' を追加
                df = pd.read_csv(file_path, encoding='cp932')
            except UnicodeDecodeError:
                # もしUTF-8だった場合のフォールバック
                df = pd.read_csv(file_path, encoding='utf-8')
            
            # 日本語カラムがあれば英語に変換
            df = df.rename(columns=column_map)
            
            # APIデータがあれば結合して保存
            if new_data_df is not None and not new_data_df.empty:
                df = pd.concat([df, new_data_df]).drop_duplicates(subset=['Date'], keep='last').reset_index(drop=True)
                df.to_csv(file_path, index=False)
        else:
            print(f"[{symbol}] データが存在しません（{file_path}）。スキップします。")
            continue

        # 3. データ形式と量のチェック
        if not all(col in df.columns for col in required_columns):
            print(f"[{symbol}] カラム構成が不足しています。期待: {required_columns}")
            continue
            
        if len(df) < 100:
            print(f"[{symbol}] データ不足です（現在 {len(df)}件）。最低100件必要です。スキップします。")
            continue
        
        print(f"[{symbol}] {len(df)}件のデータで学習を開始します...")

        # 4. 特徴量計算とOptuna最適化
        df = add_features(df)
        ranked_features = get_dynamic_features_ranked(weeks_back=4)

        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, df, ranked_features), n_trials=30)
        best_params = study.best_params
        
        # 5. 最終モデルの構築
        temp_df = df.copy()
        future_return = temp_df['Close'].shift(-30) / temp_df['Close'] - 1
        rolling_thresh = future_return.rolling(window=best_params['rolling_window']).quantile(best_params['target_quantile'])
        temp_df['Target_Long'] = (future_return > rolling_thresh).astype(int)
        temp_df = temp_df.dropna().reset_index(drop=True)

        selected_features = ranked_features[:best_params['top_n_features']]
        X_train = temp_df[selected_features]
        y_train = temp_df['Target_Long']

        lgb_params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1,
            'boosting_type': 'gbdt', 'learning_rate': best_params['learning_rate'],
            'num_leaves': best_params['num_leaves']
        }
        final_model = lgb.train(lgb_params, lgb.Dataset(X_train, label=y_train))
        
        # 6. 保存と通知
        latest_path = os.path.join(Config.MODEL_PATH, f"latest_{symbol}.joblib")
        joblib.dump({
            'model': final_model, 
            'features': selected_features,
            'atr_multiplier': best_params['atr_multiplier']
        }, latest_path)
        
        send_line_notification(f"🤖 【{symbol} 学習完了】\nスコア: {study.best_value:.4f}")
        print(f"[{symbol}] モデル保存完了: {latest_path}")
        time.sleep(3) # API制限回避

    print("\n=== 全銘柄の処理が完了しました ===")

if __name__ == "__main__":
    main()