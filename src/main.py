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
from src.train import objective, get_dynamic_features_ranked, save_feature_importance
from src.utils.notifier import send_line_notification

def main():
    print("=== AutoDayTrade マルチシンボル学習システム起動 ===")
    fetcher = KabuDataFetcher()
    os.makedirs(Config.MODEL_PATH, exist_ok=True)
    
    for symbol in Config.SYMBOLS:
        print(f"\n--- 銘柄 [{symbol}] の学習プロセスを開始 ---")
        
        # 1. データ取得
        new_data_df = fetcher.fetch_historical_data(symbol, Config.EXCHANGE)
        file_path = os.path.join(Config.RAW_DATA_PATH, f"{symbol}_history.csv")
        
        if new_data_df is None or new_data_df.empty:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
            else:
                print(f"[{symbol}] のデータが存在しません。スキップします。")
                continue
        else:
            df = fetcher.save_and_merge_data(new_data_df, symbol)

        # 2. 特徴量計算
        df = add_features(df)
        if len(df) < 50:
            print(f"[{symbol}] データ不足のためスキップします。")
            continue

        # 3. 動的特徴量ランキング取得 (銘柄ごとにログを分ける処理を内部でしている想定)
        ranked_features = get_dynamic_features_ranked(weeks_back=4)

        # 4. Optuna最適化
        print(f"[{symbol}] Optuna最適化中...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, df, ranked_features), n_trials=30)
        best_params = study.best_params
        
        # 5. 最終モデルの作成
        temp_df = df.copy()
        future_return = temp_df['Close'].shift(-30) / temp_df['Close'] - 1
        rolling_thresh = future_return.rolling(window=best_params['rolling_window']).quantile(best_params['target_quantile'])
        temp_df['Target_Long'] = (future_return > rolling_thresh).astype(int)
        temp_df = temp_df.dropna().reset_index(drop=True)

        train_size = int(len(temp_df) * 0.7)
        train_df = temp_df.iloc[:train_size]
        
        top_n = best_params['top_n_features']
        selected_features = ranked_features[:top_n]
        
        X_train, y_train = train_df[selected_features], train_df['Target_Long']

        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'learning_rate': best_params['learning_rate'],
            'num_leaves': best_params['num_leaves']
        }
        final_model = lgb.train(lgb_params, lgb.Dataset(X_train, label=y_train))
        
        # 6. モデルの保存（銘柄名を含める）
        latest_path = os.path.join(Config.MODEL_PATH, f"latest_{symbol}.joblib")
        joblib.dump({
            'model': final_model, 
            'features': selected_features,
            'atr_multiplier': best_params['atr_multiplier'] # Bot側で使うため保存
        }, latest_path)
        
        print(f"[{symbol}] モデル保存完了: {latest_path}")
        
        # LINE通知
        msg = f"🤖 【{symbol} 学習完了】\nスコア: {study.best_value:.4f}\n採用指標: {top_n}個"
        send_line_notification(msg)
        
        # API制限回避のためのインターバル
        time.sleep(3)

    print("\n=== 全銘柄の再学習が完了しました ===")

if __name__ == "__main__":
    main()