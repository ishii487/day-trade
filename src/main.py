def main():
    print("--- 蓄積・学習・最適化フェーズを開始します ---")
    
    fetcher = KabuDataFetcher()
    symbol = Config.SYMBOL
    
    # 1. 最新データの取得
    print(f"最新の分足データを取得中...")
    new_data_df = fetcher.fetch_historical_data(symbol, Config.EXCHANGE)
    
    if new_data_df is None or new_data_df.empty:
        print("最新データの取得に失敗しました。既存のCSVのみで続行します。")
        # CSVだけでも読み込む処理へ
        file_path = os.path.join(Config.RAW_DATA_PATH, f"{symbol}_history.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            print("学習用データがどこにもありません。終了します。")
            return
    else:
        # 2. 蓄積データと結合・保存
        df = fetcher.save_and_merge_data(new_data_df, symbol)

    # 3. 特徴量の付与
    print("特徴量を計算中...")
    df = add_features(df)
    
    # データが少ない場合のガード
    if len(df) < 50:
        print(f"データ数が不足しています（現在 {len(df)}件）。学習には最低50〜100件程度を推奨します。")
        # 試運転を続行する場合はここで return せずに進む
    
    # 4. 最適化と学習
    print(f"過去 {len(df)} 件のデータを用いて最適化を開始します...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, df), n_trials=30)
    
    print("--- 最適化完了 ---")
    print(f"最良スコア: {study.best_value}")
    print(f"最良パラメータ: {study.best_params}")

if __name__ == "__main__":
    main()