import os
import pandas as pd
import optuna
import lightgbm as lgb
import joblib
from datetime import datetime
from src.config import Config
from src.data_fetcher import KabuDataFetcher
from src.features import add_features
from src.train import objective, get_dynamic_features_ranked, save_feature_importance
from src.reporter import generate_report

def main():
    print("=== 自動デイトレード・自律学習システム起動 ===")
    
    fetcher = KabuDataFetcher()
    symbol = Config.SYMBOL
    
    # ---------------------------------------------------------
    # 1. データの取得と蓄積
    # ---------------------------------------------------------
    print("\n[1/5] 最新データの取得と蓄積を開始...")
    new_data_df = fetcher.fetch_historical_data(symbol, Config.EXCHANGE)
    
    file_path = os.path.join(Config.RAW_DATA_PATH, f"{symbol}_history.csv")
    if new_data_df is None or new_data_df.empty:
        print("最新データの取得に失敗しました。既存のCSVを読み込みます。")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            print("学習用データが存在しません。処理を終了します。")
            return
    else:
        df = fetcher.save_and_merge_data(new_data_df, symbol)

    # ---------------------------------------------------------
    # 2. 特徴量の付与
    # ---------------------------------------------------------
    print("\n[2/5] テクニカル指標（特徴量）の計算中...")
    df = add_features(df)
    
    if len(df) < 50:
        print(f"データが不足しています（現在 {len(df)}件）。学習をスキップします。")
        return

    # ---------------------------------------------------------
    # 3. 動的特徴量のランキング取得
    # ---------------------------------------------------------
    print("\n[3/5] 過去の成績に基づく特徴量ランキングの取得...")
    ranked_features = get_dynamic_features_ranked(weeks_back=4)

    # ---------------------------------------------------------
    # 4. Optunaによる最適化（シミュレーション）
    # ---------------------------------------------------------
    print("\n[4/5] Optunaによる戦略の最適化を開始します...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, df, ranked_features), n_trials=30)
    
    best_params = study.best_params
    print(f"\n--- 最適化完了 ---")
    print(f"最良期待値（スコア）: {study.best_value}")
    print(f"最良パラメータ: {best_params}")

    # ---------------------------------------------------------
    # 5. 本番用モデルの作成と重要度ログの保存
    # ---------------------------------------------------------
    print("\n[5/5] 最良パラメータを用いた本番用モデルの構築...")
    
    # ターゲット変数の再作成（best_paramsを使用）
    temp_df = df.copy()
    future_return = temp_df['Close'].shift(-30) / temp_df['Close'] - 1
    rolling_thresh = future_return.rolling(window=best_params['rolling_window']).quantile(best_params['target_quantile'])
    temp_df['Target_Long'] = (future_return > rolling_thresh).astype(int)
    temp_df = temp_df.dropna().reset_index(drop=True)

    # 学習データの準備
    train_size = int(len(temp_df) * 0.7)
    train_df = temp_df.iloc[:train_size]
    
    top_n = best_params['top_n_features']
    selected_features = ranked_features[:top_n]
    
    X_train = train_df[selected_features]
    y_train = train_df['Target_Long']

    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': best_params['learning_rate'],
        'num_leaves': best_params['num_leaves']
    }

    # 最終学習
    final_model = lgb.train(lgb_params, lgb.Dataset(X_train, label=y_train))
    
    final_preds = final_model.predict(X_valid)
    final_signals = (final_preds > 0.5).astype(int)
    
    final_backtest_results = calculate_trailing_stop_returns(
        final_signals, 
        valid_df['High'].values, 
        valid_df['Low'].values, 
        valid_df['Close'].values, 
        valid_df['ATR'].values, 
        best_params['atr_multiplier']
    )

    # 保存処理とレポート生成の呼び出し
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    save_feature_importance(final_model, selected_features)
    save_model(final_model, study.best_value, selected_features)  # joblibでの保存
    
    # 新しいレポーター関数の呼び出し
    generate_report(final_model, selected_features, final_backtest_results, timestamp, study.best_value)
    
    print("\n=== 全プロセスが正常に完了しました ===")

def save_model(model, score, features):
    """モデルを日時とスコア付きで保存し、latestとしても更新する"""
    os.makedirs("models", exist_ok=True)
    
    # 1. 刻印用ファイル名の作成 (例: model_20260316_1300_S025.joblib)
    # スコアは小数点以下3桁まで表示
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    formatted_score = f"{score:.3f}".replace(".", "")
    filename = f"model_{timestamp}_S{formatted_score}.joblib"
    
    save_path = os.path.join("models", filename)
    latest_path = os.path.join("models", "latest.joblib")
    
# モデル本体と、そのモデルが使う特徴量のリストをセットにして保存
    data_to_save = {
        'model': model,
        'features': features
    }
    joblib.dump(data_to_save, save_path)
    joblib.dump(data_to_save, latest_path)
    print("最新モデルと特徴量リストを保存しました。")

if __name__ == "__main__":
    main()