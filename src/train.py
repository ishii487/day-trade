import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from datetime import datetime, timedelta
# バックテスト関数のインポートは現状維持
from src.backtest import calculate_trailing_stop_returns 

FI_LOG_PATH = "data/processed/feature_importance_log.csv"

# 【修正1】デフォルト特徴量リストを相対化カラム名に更新
DEFAULT_FEATURES = [
    'Return_1m', 'Return_5m', 'Return_15m', 
    'SMA_15_Dev', 'SMA_60_Dev', 
    'ATR_Pct', 'Volume_Ratio', 'RSI_14'
]

def get_dynamic_features_ranked(weeks_back=4):
    """過去のログから特徴量を重要度の降順でソートしたリストを返す"""
    if not os.path.exists(FI_LOG_PATH):
        # ログがない場合は、新しいデフォルトリストを返す
        return DEFAULT_FEATURES
        
    try:
        df_fi = pd.read_csv(FI_LOG_PATH)
        df_fi['date'] = pd.to_datetime(df_fi['date'])
        
        # 過去指定週間のデータを抽出
        cutoff = datetime.now() - timedelta(weeks=weeks_back)
        recent_fi = df_fi[df_fi['date'] >= cutoff].drop(columns=['date'])
        
        if len(recent_fi) == 0:
            return DEFAULT_FEATURES
            
        # 平均重要度を計算し、降順に並べ替え
        mean_fi = recent_fi.mean().sort_values(ascending=False)
        ranked_features = mean_fi[mean_fi > 0].index.tolist()
        
        # 実際にデータフレームに存在するカラムのみにフィルタリング（セーフティ）
        ranked_features = [f for f in ranked_features if f in DEFAULT_FEATURES]

        if len(ranked_features) < 3:
            return DEFAULT_FEATURES
            
        print(f"過去{weeks_back}週間の実績から、以下の順で特徴量を評価します:\n{ranked_features}")
        return ranked_features
    except:
        return DEFAULT_FEATURES

def save_feature_importance(model, features):
    """最終モデルの特徴量重要度をCSVに追記保存する"""
    importance = model.feature_importance(importance_type='gain')
    row_data = {'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    row_data.update({feat: imp for feat, imp in zip(features, importance)})
    df_new = pd.DataFrame([row_data])
    os.makedirs(os.path.dirname(FI_LOG_PATH), exist_ok=True)
    
    if os.path.exists(FI_LOG_PATH):
        df_new.to_csv(FI_LOG_PATH, mode='a', header=False, index=False)
    else:
        df_new.to_csv(FI_LOG_PATH, index=False)
    print(f"特徴量重要度を記録しました。({FI_LOG_PATH})")

def objective(trial, df, ranked_features):
    """Optunaが実行する評価関数"""
    top_n = trial.suggest_int('top_n_features', 3, len(ranked_features))
    selected_features = ranked_features[:top_n]
    
    rolling_window = trial.suggest_int('rolling_window', 20, 200)
    target_quantile = trial.suggest_float('target_quantile', 0.5, 0.95)
    atr_multiplier = trial.suggest_float('atr_multiplier', 1.5, 5.0)
    
    temp_df = df.copy()
    
    # ターゲット変数の作成（ここはCloseの相対変化なのでOK）
    future_return = temp_df['Close'].shift(-30) / temp_df['Close'] - 1
    rolling_thresh = future_return.rolling(window=rolling_window).quantile(target_quantile)
    temp_df['Target_Long'] = (future_return > rolling_thresh).astype(int)
    
    temp_df = temp_df.dropna().reset_index(drop=True)
    if len(temp_df) < 50:
        return -100.0
        
    train_size = int(len(temp_df) * 0.7)
    train_df = temp_df.iloc[:train_size]
    valid_df = temp_df.iloc[train_size:]
    
    X_train, y_train = train_df[selected_features], train_df['Target_Long']
    X_valid, y_valid = valid_df[selected_features], valid_df['Target_Long']
    
    lgb_params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
    }
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)
    model = lgb.train(lgb_params, dtrain, valid_sets=[dvalid])
    
    preds = model.predict(X_valid)
    signals = (preds > 0.5).astype(int)
    
    # 【修正2】バックテストに渡すATRを「相対化ATR（ATR_Pct）」に合わせるか、
    # 既存の計算結果が残っている場合はそれを使用
    # ここでは features.py で計算している ATR_Pct を活用します
    backtest_results = calculate_trailing_stop_returns(
        signals, 
        valid_df['High'].values, 
        valid_df['Low'].values, 
        valid_df['Close'].values, 
        valid_df['ATR_Pct'].values * valid_df['Close'].values, # 絶対値に戻して計算
        atr_multiplier
    )
    
    total_return = backtest_results['total_return']
    max_drawdown = backtest_results['max_drawdown']
    
    epsilon = 1e-6
    if total_return <= 0:
        score = total_return
    else:
        score = total_return / (abs(max_drawdown) + epsilon)
    
    return score