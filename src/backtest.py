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
def calculate_trailing_stop_returns(signals, high, low, close, atr, atr_multiplier):
    """
    シグナルとトレイリングストップに基づくバックテストを実行し、詳細な統計を辞書で返す
    """
    equity = 1.0  # 初期資金を1（100%）とする
    equity_curve = [equity]
    trade_returns = []
    
    winning_trades = 0
    losing_trades = 0
    in_position = False
    entry_price = 0.0
    stop_loss = 0.0

    # シグナルの配列をループしてトレードをシミュレーション
    for i in range(len(signals) - 1):
        if not in_position and signals[i] == 1:
            # エントリー（買い）
            in_position = True
            entry_price = close[i]
            # 初期ストップロスの設定
            stop_loss = entry_price - (atr[i] * atr_multiplier)
        
        elif in_position:
            # トレイリングストップの切り上げ
            current_stop = close[i] - (atr[i] * atr_multiplier)
            if current_stop > stop_loss:
                stop_loss = current_stop
            
            # 損切り・利確の判定（安値がストップロスに触れたか）
            if low[i] <= stop_loss:
                in_position = False
                # 決済価格はストップロス価格とする（スリッページは一旦考慮しない）
                trade_return = (stop_loss - entry_price) / entry_price
                trade_returns.append(trade_return)
                equity *= (1 + trade_return)
                
                if trade_return > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1
                    
        # 毎ステップの資産を記録
        equity_curve.append(equity)
        
    # バックテスト終了時にポジションを持っていた場合の時価評価（強制決済）
    if in_position:
        trade_return = (close[-1] - entry_price) / entry_price
        trade_returns.append(trade_return)
        equity *= (1 + trade_return)
        if trade_return > 0:
            winning_trades += 1
        else:
            losing_trades += 1
        equity_curve.append(equity)

    # ---------------------------------------------------------
    # 統計データの計算
    # ---------------------------------------------------------
    total_trades = winning_trades + losing_trades
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    
    # 最大ドローダウン（Max Drawdown）の計算
    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.cummax()  # 過去の最高資産を保持
    drawdowns = (equity_series - rolling_max) / rolling_max  # 最高値からの下落率
    max_drawdown = drawdowns.min()  # マイナス方向の最大値

    # レポートと最適化に必要な全データを返す
    return {
        'total_return': equity - 1.0,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'equity_curve': equity_curve
    }
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