import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def generate_report(model, selected_features, backtest_results, timestamp, score):
    """
    モデルの特徴量重要度と、詳細な統計データ（勝率・DD等）を含むバックテスト結果を画像として保存する
    """
    os.makedirs("reports", exist_ok=True)
    
    # ファイル名のベース作成
    base_name = f"report_{timestamp}_S{f'{score:.3f}'.replace('.', '')}"
    
    # ---------------------------------------------------------
    # 1. 特徴量重要度のグラフ化
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    importance = model.feature_importance(importance_type='gain')
    fi_df = pd.DataFrame({'feature': selected_features, 'importance': importance})
    fi_df = fi_df.sort_values('importance', ascending=False)
    
    sns.barplot(x='importance', y='feature', data=fi_df, palette='viridis')
    plt.title(f"Feature Importance (Gain) - Score: {score:.4f}")
    plt.tight_layout()
    plt.savefig(f"reports/{base_name}_importance.png")
    plt.close()

    # ---------------------------------------------------------
    # 2. 資産推移のグラフ化（詳細な統計データ印字）
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # 資産推移曲線のプロット
    equity_curve = backtest_results['equity_curve']
    plt.plot(equity_curve, label='Strategy Equity', color='blue')
    plt.axhline(1.0, color='red', linestyle='--', alpha=0.5) # 初期資金(1.0)のライン
    
    # 印字する統計テキストの作成
    total_return_pct = backtest_results['total_return'] * 100
    win_rate_pct = backtest_results['win_rate'] * 100
    max_dd_pct = backtest_results['max_drawdown'] * 100
    
    stats_text = (
        f"--- Backtest Results ---\n"
        f"Total Return: {total_return_pct:.2f}%\n"
        f"Win Rate: {win_rate_pct:.2f}%\n"
        f"Max Drawdown: {max_dd_pct:.2f}%\n"
        f"Total Trades: {backtest_results['total_trades']}\n"
        f"Risk-Adj Score: {score:.4f}"
    )
    
    # グラフの左上にテキストボックスを配置
    plt.gca().text(
        0.02, 0.96, stats_text, 
        transform=plt.gca().transAxes,
        fontsize=11, 
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    )

    plt.title(f"Backtest Profit Curve & Stats - {timestamp}")
    plt.xlabel("Trade Count / Steps")
    plt.ylabel("Equity (1.0 = Initial Capital)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"reports/{base_name}_performance.png")
    plt.close()

    print(f"詳細レポートを生成しました: reports/{base_name}_performance.png")