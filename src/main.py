import os
import glob
import joblib
import pandas as pd
import optuna
import lightgbm as lgb
from src.config import Config
from src.features import add_features
from src.train import objective, get_dynamic_features_ranked
from src.utils.notifier import send_line_notification

def main():
    print("=== AutoDayTrade 業界別・汎用モデル学習システム ===")
    os.makedirs(Config.MODEL_PATH, exist_ok=True)
    
    # 業界フォルダのリストを取得（data/raw/ 以下のディレクトリ）
    sector_dirs = [d for d in glob.glob(os.path.join(Config.RAW_DATA_PATH, "*")) if os.path.isdir(d)]

    for sector_path in sector_dirs:
        sector_name = os.path.basename(sector_path)
        print(f"\n--- 業界 [{sector_name}] の一括学習を開始 ---")
        
        # 1. フォルダ内の全CSVを読み込んで合体させる
        all_files = glob.glob(os.path.join(sector_path, "*.csv"))
        if not all_files:
            print(f"[{sector_name}] データファイルが見つかりません。スキップします。")
            continue
            
        sector_df_list = []
        for file in all_files:
            # Shift-JIS(cp932)とUTF-8の両方に対応
            try:
                temp_df = pd.read_csv(file, encoding='cp932')
            except:
                temp_df = pd.read_csv(file, encoding='utf-8')
            
            # カラム名の正規化（以前作成したマッピングを使用）
            temp_df = temp_df.rename(columns={
                '日付': 'Date', '日時': 'Date', '始値': 'Open', '寄付': 'Open',
                '高値': 'High', '安値': 'Low', '終値': 'Close', '出来高': 'Volume'
            })
            
            # 相対化特徴量を計算（銘柄ごとに計算してから合体させるのが重要）
            temp_df = add_features(temp_df)
            sector_df_list.append(temp_df)
            print(f"  - {os.path.basename(file)} を読み込みました ({len(temp_df)}行)")

        # 全銘柄を合体
        df = pd.concat(sector_df_list, ignore_index=True)
        print(f"[{sector_name}] 合計 {len(df)}行 のビッグデータで学習を開始します...")

        # 2. Optuna最適化
        ranked_features = get_dynamic_features_ranked()
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, df, ranked_features), n_trials=30)
        
        # 3. 最終モデルの構築と保存
        best_params = study.best_params
        selected_features = ranked_features[:best_params['top_n_features']]
        
        # ターゲット再生成（最終学習用）
        future_return = df['Close'].shift(-30) / df['Close'] - 1
        rolling_thresh = future_return.rolling(window=best_params['rolling_window']).quantile(best_params['target_quantile'])
        df['Target_Long'] = (future_return > rolling_thresh).astype(int)
        df = df.dropna().reset_index(drop=True)

        final_model = lgb.train(
            {'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1, 'num_leaves': best_params['num_leaves']},
            lgb.Dataset(df[selected_features], label=df['Target_Long'])
        )
        
        # 保存：銘柄名ではなく業界名で保存
        model_name = f"sector_{sector_name}.joblib"
        save_path = os.path.join(Config.MODEL_PATH, model_name)
        joblib.dump({
            'model': final_model, 
            'features': selected_features,
            'atr_multiplier': best_params['atr_multiplier']
        }, save_path)
        
        print(f"[{sector_name}] 汎用モデル保存完了: {save_path}")

    print("\n=== 全業界の汎用モデル構築が完了しました ===")

if __name__ == "__main__":
    main()