import talib
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

# 保存先
FI_LOG_PATH = "data/processed/feature_importance_log.csv"

def add_features(df):
    """
    株価の絶対値ではなく、相対的な変化率（%）や倍率に変換した特徴量を生成する
    """
    df = df.copy()

    # 1. 相対的な価格変化率（リターン）
    df['Return_1m'] = df['Close'].pct_change(1)
    df['Return_5m'] = df['Close'].pct_change(5)
    df['Return_15m'] = df['Close'].pct_change(15)

    # 2. 移動平均からの乖離率（パーセント）
    df['SMA_15'] = df['Close'].rolling(window=15).mean()
    df['SMA_15_Dev'] = (df['Close'] / df['SMA_15']) - 1.0  # +0.01 なら1%の上方乖離

    df['SMA_60'] = df['Close'].rolling(window=60).mean()
    df['SMA_60_Dev'] = (df['Close'] / df['SMA_60']) - 1.0

    # 3. ボラティリティの正規化（ATRを株価で割ってパーセント化）
    df['TR'] = np.maximum((df['High'] - df['Low']),
               np.maximum(abs(df['High'] - df['Close'].shift(1)),
                          abs(df['Low'] - df['Close'].shift(1))))
    df['ATR_14'] = df['TR'].rolling(window=14).mean()
    df['ATR_Pct'] = df['ATR_14'] / df['Close'] # 株価に対するボラティリティの割合

    # 4. 出来高の急増度合い（過去15分平均に対する倍率）
    df['Vol_SMA_15'] = df['Volume'].rolling(window=15).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Vol_SMA_15'].replace(0, np.nan) # 3.0なら平均の3倍の出来高

    # 5. RSI (元々0〜100に正規化されているオシレーター指標なのでそのまま使用)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # 💡 AIのカンニング・ノイズ防止：絶対値を含む計算用カラムを削除する
    # Open, High, Low, Close自体はターゲット生成に使うため残すが、学習特徴量からは外れる設計にする
    df = df.drop(columns=['SMA_15', 'SMA_60', 'TR', 'ATR_14', 'Vol_SMA_15'])

    # 欠損値（計算不能な最初の数行）を削除
    df = df.dropna().reset_index(drop=True)

    return df

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