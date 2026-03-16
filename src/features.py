import talib
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

# 保存先
FI_LOG_PATH = "data/processed/feature_importance_log.csv"

def add_features(df):
    """
    TA-Libを用いて多角的な特徴量を追加する
    """
    # 念のためDataFrameのコピーを作成
    df = df.copy()

    # 1. ボラティリティ
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(
        df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    
    # 2. モメンタム
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'])
    df['STOCH_k'], df['STOCH_d'] = talib.STOCH(
        df['High'], df['Low'], df['Close'], 
        fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0
    )

    # 3. トレンド
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)

    # 4. 出来高
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])

    # 5. リターン（過去の変化率）
    df['Return_1'] = df['Close'].pct_change(1)
    df['Return_5'] = df['Close'].pct_change(5)
    df['Return_15'] = df['Close'].pct_change(15)
    df['Return_30'] = df['Close'].pct_change(30)

    # 欠損値を削除
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