import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import talib
from dotenv import load_dotenv

# ---------------------------------------------------
# 1. セキュリティ設定：環境変数の読み込み
# ---------------------------------------------------
load_dotenv()
KABU_API_PASSWORD = os.getenv("KABU_API_PASSWORD")
KABU_API_PORT = os.getenv("KABU_API_PORT")

if not KABU_API_PASSWORD:
    raise ValueError("環境変数 KABU_API_PASSWORD が設定されていません。.envファイルを確認してください。")

# ---------------------------------------------------
# 2. 高速バックテスト関数（ATRトレイリングストップ）
# ---------------------------------------------------
def calculate_trailing_stop_returns(signals, highs, lows, closes, atrs, atr_multiplier):
    """
    シグナルとATRトレイリングストップを用いたバックテストを行い、総利益率を返す
    """
    returns = []
    in_position = False
    entry_price = 0.0
    highest_price = 0.0
    stop_loss = 0.0
    
    # Optuna内で高速に回すため、Numpy配列でループ処理
    for i in range(len(signals)):
        if not in_position:
            if signals[i] == 1:  # エントリーシグナル発生（ロング）
                in_position = True
                # 簡単のためシグナル発生足の終値でエントリーと仮定
                # （より厳密には次足の始値とします）
                entry_price = closes[i]
                highest_price = entry_price
                stop_loss = highest_price - (atrs[i] * atr_multiplier)
        else:
            # ポジション保有中：最高値の更新とトレイリングストップの切り上げ
            if highs[i] > highest_price:
                highest_price = highs[i]
                # ストップラインを下げることはせず、高い方のみを採用
                stop_loss = max(stop_loss, highest_price - (atrs[i] * atr_multiplier))
            
            # 安値がストップラインに触れたか判定（損切り・利確の実行）
            if lows[i] <= stop_loss:
                # 厳密にはギャップダウンも考慮し、min(始値, stop_loss)等にするのが安全
                exit_price = stop_loss 
                trade_return = (exit_price - entry_price) / entry_price
                returns.append(trade_return)
                in_position = False
                
    # 取引が1回も発生しなかった場合はペナルティとしてマイナス値を返す
    if len(returns) == 0:
        return -1.0
        
    # 今回は「総利益率（合計リターン）」を評価指標とする
    # ※ドローダウンを嫌う場合は、シャープレシオなどに変更可能
    return np.sum(returns)

# ---------------------------------------------------
# 3. Optuna 目的関数（パイプライン）
# ---------------------------------------------------
def objective(trial, df_historical):
    df = df_historical.copy()
    
    # パラメータの探索（マジックナンバーの排除）
    rolling_window = trial.suggest_int('rolling_window', 10, 100)
    target_quantile = trial.suggest_float('target_quantile', 0.6, 0.95)
    atr_multiplier = trial.suggest_float('atr_multiplier', 1.0, 5.0)
    
    # 特徴量生成（TA-Lib）
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    # ※ここにMACDやRSIなど他の特徴量を追加
    
    # 動的ターゲット変数の生成（n分後のリターンが上位X%に入るか）
    future_return = df['Close'].shift(-1) / df['Close'] - 1
    rolling_thresh = future_return.rolling(window=rolling_window).quantile(target_quantile)
    df['Target_Long'] = (future_return > rolling_thresh).astype(int)
    
    df = df.dropna()
    
    # 学習用と検証用に分割（時系列のため単純なランダム分割はNG）
    train_size = int(len(df) * 0.7)
    train_df = df.iloc[:train_size]
    valid_df = df.iloc[train_size:]
    
    features = ['ATR'] # 実際にはここに特徴量リストを指定
    X_train, y_train = train_df[features], train_df['Target_Long']
    X_valid, y_valid = valid_df[features], valid_df['Target_Long']
    
    # LightGBMの学習
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'verbose': -1
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
    # Early stopping等も組み込み可能
    model = lgb.train(lgb_params, train_data, valid_sets=[valid_data])
    
    # 検証データに対する予測（確率）
    preds_prob = model.predict(X_valid)
    
    # 確率を0 or 1のシグナルに変換（ここでは簡易的に0.5を閾値とする）
    # ※この0.5自体もOptunaで最適化することが可能です
    signals = (preds_prob > 0.5).astype(int)
    
    # バックテスト評価関数に渡すためにNumpy配列を抽出
    highs = valid_df['High'].values
    lows = valid_df['Low'].values
    closes = valid_df['Close'].values
    atrs = valid_df['ATR'].values
    
    # 最終的な期待値（総利益率）を計算
    total_return = calculate_trailing_stop_returns(
        signals, highs, lows, closes, atrs, atr_multiplier
    )
    
    # Optunaは戻り値を最大化するように設定（direction='maximize'）
    return total_return