import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from datetime import datetime, timedelta
from src.backtest import calculate_trailing_stop_returns

FI_LOG_PATH = "data/processed/feature_importance_log.csv"

# features.pyで作成される全特徴量のリスト（デフォルト）
DEFAULT_FEATURES = [
    'ATR', 'BB_width', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
    'STOCH_k', 'STOCH_d', 'ADX', 'OBV',
    'Return_1', 'Return_5', 'Return_15', 'Return_30'
]

def get_dynamic_features_ranked(weeks_back=4):
    """過去のログから特徴量を重要度（平均）の降順でソートしたリストを返す"""
    if not os.path.exists(FI_LOG_PATH):
        print("重要度ログが存在しないため、デフォルトの特徴量リストを使用します。")
        return DEFAULT_FEATURES
        
    df_fi = pd.read_csv(FI_LOG_PATH)
    df_fi['date'] = pd.to_datetime(df_fi['date'])
    
    # 過去指定週間のデータを抽出
    cutoff = datetime.now() - timedelta(weeks=weeks_back)
    recent_fi = df_fi[df_fi['date'] >= cutoff].drop(columns=['date'])
    
    if len(recent_fi) == 0:
        return DEFAULT_FEATURES
        
    # 平均重要度を計算し、降順（大きい順）に並べ替え
    mean_fi = recent_fi.mean().sort_values(ascending=False)
    
    # スコアが0より大きいものをリスト化
    ranked_features = mean_fi[mean_fi > 0].index.tolist()
    
    # 念のためのセーフティネット
    if len(ranked_features) < 3:
        return DEFAULT_FEATURES
        
    print(f"過去{weeks_back}週間の実績から、以下の順で特徴量を評価します:\n{ranked_features}")
    return ranked_features

def save_feature_importance(model, features):
    """最終モデルの特徴量重要度をCSVに追記保存する"""
    importance = model.feature_importance(importance_type='gain')
    
    # 記録用フォーマットの作成
    row_data = {'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    row_data.update({feat: imp for feat, imp in zip(features, importance)})
    df_new = pd.DataFrame([row_data])
    
    # CSV保存ディレクトリの確保
    os.makedirs(os.path.dirname(FI_LOG_PATH), exist_ok=True)
    
    if os.path.exists(FI_LOG_PATH):
        df_new.to_csv(FI_LOG_PATH, mode='a', header=False, index=False)
    else:
        df_new.to_csv(FI_LOG_PATH, index=False)
    print(f"特徴量重要度を記録しました。({FI_LOG_PATH})")

def objective(trial, df, ranked_features):
    """Optunaが実行する評価関数"""
    
    # 【マジックナンバー排除】上位何個の特徴量を使用するかをOptunaに決定させる
    # 最小3個 〜 最大(ランキングの総数) の間で探索
    top_n = trial.suggest_int('top_n_features', 3, len(ranked_features))
    selected_features = ranked_features[:top_n]
    
    # その他のハイパーパラメータ
    rolling_window = trial.suggest_int('rolling_window', 20, 200)
    target_quantile = trial.suggest_float('target_quantile', 0.5, 0.95)
    atr_multiplier = trial.suggest_float('atr_multiplier', 1.5, 5.0)
    
    temp_df = df.copy()
    
    # 動的ターゲット変数の作成
    future_return = temp_df['Close'].shift(-30) / temp_df['Close'] - 1
    rolling_thresh = future_return.rolling(window=rolling_window).quantile(target_quantile)
    temp_df['Target_Long'] = (future_return > rolling_thresh).astype(int)
    
    temp_df = temp_df.dropna().reset_index(drop=True)
    if len(temp_df) < 50:
        return -100.0 # データ不足ペナルティ
        
    # 時系列分割（7:3）
    train_size = int(len(temp_df) * 0.7)
    train_df = temp_df.iloc[:train_size]
    valid_df = temp_df.iloc[train_size:]
    
    # 選択された特徴量のみを抽出
    X_train, y_train = train_df[selected_features], train_df['Target_Long']
    X_valid, y_valid = valid_df[selected_features], valid_df['Target_Long']
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
    }
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)
    
    # 学習
    model = lgb.train(lgb_params, dtrain, valid_sets=[dvalid])
    
    # 予測とバックテスト評価
    preds = model.predict(X_valid)
    signals = (preds > 0.5).astype(int)
    
    total_return = calculate_trailing_stop_returns(
        signals, 
        valid_df['High'].values, 
        valid_df['Low'].values, 
        valid_df['Close'].values, 
        valid_df['ATR'].values, 
        atr_multiplier
    )
    
    return total_return