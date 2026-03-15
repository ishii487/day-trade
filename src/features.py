import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# 保存先
FI_LOG_PATH = "data/processed/feature_importance_log.csv"

def save_feature_importance(model, features):
    """
    学習完了後に、LightGBMのモデルから特徴量重要度を取得しCSVに追記保存する
    """
    # importance_type='gain' は、その特徴量がどれだけ分岐の精度向上に貢献したかを示します
    importance = model.feature_importance(importance_type='gain')
    
    # 記録用のデータフレームを作成
    df_fi = pd.DataFrame({
        'date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        **{feat: [imp] for feat, imp in zip(features, importance)}
    })
    
    # CSVに追記（ファイルがなければ新規作成）
    if os.path.exists(FI_LOG_PATH):
        df_fi.to_csv(FI_LOG_PATH, mode='a', header=False, index=False)
    else:
        df_fi.to_csv(FI_LOG_PATH, index=False)
        
    print("特徴量重要度のログを保存しました。")

def get_dynamic_features(all_features, weeks_back=4, top_n=8):
    """
    過去数週間分の重要度ログを読み込み、平均スコアが高い上位N個の特徴量を返す
    """
    if not os.path.exists(FI_LOG_PATH):
        # ログが存在しない初回実行時は、とりあえず全ての特徴量を使用する
        print("重要度ログが存在しないため、全ての特徴量を使用します。")
        return all_features
        
    df_fi = pd.read_csv(FI_LOG_PATH)
    df_fi['date'] = pd.to_datetime(df_fi['date'])
    
    # 過去指定された週間分のデータを抽出
    cutoff_date = datetime.now() - timedelta(weeks=weeks_back)
    recent_fi = df_fi[df_fi['date'] >= cutoff_date].copy()
    
    if len(recent_fi) == 0:
        return all_features
        
    # 日付カラムを除外して、各特徴量の平均重要度を計算
    mean_importance = recent_fi.drop(columns=['date']).mean()
    
    # スコアが0のもの（全く使われなかったもの）を除外し、上位トップNを取得
    best_features = mean_importance[mean_importance > 0].nlargest(top_n).index.tolist()
    
    # 最低限の特徴量が確保できなかった場合のセーフティネット
    if len(best_features) < 3:
        print("有効な特徴量が少なすぎるため、デフォルトのセットを使用します。")
        return all_features
        
    print(f"過去{weeks_back}週間のデータから、以下の特徴量を選定しました: {best_features}")
    return best_features