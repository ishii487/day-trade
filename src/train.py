import lightgbm as lgb
import pandas as pd
import numpy as np
from src.backtest import calculate_trailing_stop_returns

def objective(trial, df):
    # 特徴量リストの定義（features.pyで追加したものすべて）
    features = [
        'ATR', 'BB_width', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
        'STOCH_k', 'STOCH_d', 'ADX', 'OBV',
        'Return_1', 'Return_5', 'Return_15', 'Return_30'
    ]
    
    # Optunaによるハイパーパラメータの探索
    rolling_window = trial.suggest_int('rolling_window', 20, 200)
    target_quantile = trial.suggest_float('target_quantile', 0.5, 0.95)
    atr_multiplier = trial.suggest_float('atr_multiplier', 1.5, 5.0)
    
    # 目的変数の作成（前回のfeatures.pyのロジックを適用）
    # ※ここでは簡略化のため、dfは既にadd_features済みと仮定
    future_return = df['Close'].shift(-30) / df['Close'] - 1
    rolling_thresh = future_return.rolling(window=rolling_window).quantile(target_quantile)
    df['Target_Long'] = (future_return > rolling_thresh).astype(int)
    
    # データのクレンジング
    temp_df = df.dropna().copy()
    if len(temp_df) < 50:
        return -100.0 # データ不足時はペナルティ

    # 時系列分割（7:3）
    train_size = int(len(temp_df) * 0.7)
    train_df = temp_df.iloc[:train_size]
    valid_df = temp_df.iloc[train_size:]
    
    X_train, y_train = train_df[features], train_df['Target_Long']
    X_valid, y_valid = valid_df[features], valid_df['Target_Long']
    
    # LightGBMパラメータ
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 128),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
    }
    
    # 学習
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)
    model = lgb.train(lgb_params, dtrain, valid_sets=[dvalid])
    
    # 予測とバックテストによる評価
    preds = model.predict(X_valid)
    signals = (preds > 0.5).astype(int) # 0.5以上を買いシグナルとする
    
    total_return = calculate_trailing_stop_returns(
        signals, 
        valid_df['High'].values, 
        valid_df['Low'].values, 
        valid_df['Close'].values, 
        valid_df['ATR'].values, 
        atr_multiplier
    )
    
    return total_return