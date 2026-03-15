import talib
import numpy as np
import pandas as pd

def add_features(df):
    """TA-Libを用いて特徴量を追加する"""
    # 後のOptunaで期間も調整可能ですが、まずは標準的な期間で計算
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['Close'])
    
    # 変化率（モメンタム）
    df['Return_1h'] = df['Close'].pct_change(60) # 1分足なら60本で1時間
    
    return df

def add_dynamic_target(df, window, quantile):
    """
    過去window期間の分布に基づき、上位quantile%に入るリターンを1とするターゲットを作成
    """
    # 30分後（30本先）のリターンを予測対象とする例
    df['future_return'] = df['Close'].shift(-30) / df['Close'] - 1
    
    # 動的な閾値計算（マジックナンバーの排除）
    df['rolling_thresh'] = df['future_return'].rolling(window=window).quantile(quantile)
    
    # ターゲット：将来のリターンが閾値を超えていれば1（ロング）
    df['target'] = (df['future_return'] > df['rolling_thresh']).astype(int)
    
    return df