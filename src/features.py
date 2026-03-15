import talib
import numpy as np
import pandas as pd

def add_features(df):
    """
    TA-Libを用いて多角的な特徴量（トレンド、モメンタム、ボラティリティ、出来高）を追加する
    """
    # ---------------------------------------------------------
    # 1. ボラティリティ（変動率・価格帯）
    # ---------------------------------------------------------
    # ATR (Average True Range): 既にトレイリングストップで利用中だが特徴量としても優秀
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    
    # ボリンジャーバンド: ±2シグマのバンド幅と、現在値のバンド内位置を計算
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(
        df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    # バンド幅（ボラティリティの拡大・縮小を捉えるスクイーズ/エクスパンション）
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    
    # ---------------------------------------------------------
    # 2. モメンタム（相場の勢い・買われすぎ/売られすぎ）
    # ---------------------------------------------------------
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'])
    
    # ストキャスティクス: %Kと%D
    df['STOCH_k'], df['STOCH_d'] = talib.STOCH(
        df['High'], df['Low'], df['Close'], 
        fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0
    )

    # ---------------------------------------------------------
    # 3. トレンド（方向性と強さ）
    # ---------------------------------------------------------
    # ADX (Average Directional Movement Index): トレンドの「強さ」を示す（方向は問わない）
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)

    # ---------------------------------------------------------
    # 4. 出来高（資金の流入・流出）
    # ---------------------------------------------------------
    # OBV (On Balance Volume): 上昇日の出来高を足し、下落日の出来高を引いた累積値
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])

    # ---------------------------------------------------------
    # 5. マルチタイムフレーム・リターン（過去からの変化率）
    # ---------------------------------------------------------
    # 単純な現在値ではなく、過去N本前からの「変化率」を入れることで、
    # 決定木モデルがスケール（価格の絶対値）に依存せずに学習しやすくなります。
    df['Return_1'] = df['Close'].pct_change(1)   # 1本前からの変化率
    df['Return_5'] = df['Close'].pct_change(5)   # 5本前からの変化率
    df['Return_15'] = df['Close'].pct_change(15) # 15本前からの変化率
    df['Return_30'] = df['Close'].pct_change(30) # 30本前からの変化率

    # TA-Libの計算等で発生した NaN（欠損値）を削除
    df = df.dropna().reset_index(drop=True)
    
    return df

def add_dynamic_target(df, window, quantile):
    """過去window期間の分布に基づき、上位quantile%に入るリターンを1とするターゲットを作成"""
    df['future_return'] = df['Close'].shift(-30) / df['Close'] - 1
    df['rolling_thresh'] = df['future_return'].rolling(window=window).quantile(quantile)
    df['Target_Long'] = (future_return > df['rolling_thresh']).astype(int)
    
    # 将来のリターン計算で末尾に発生する NaN を削除
    df = df.dropna().reset_index(drop=True)
    
    return df