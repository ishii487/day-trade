import os
import time
import joblib
import pandas as pd
from datetime import datetime
from src.config import Config
from src.data_fetcher import KabuDataFetcher
from src.features import add_features
from src.utils.notifier import send_line_notification

LOG_PATH = "data/processed/paper_trades.csv"

class PaperTrader:
    def __init__(self, symbols, initial_cash=1000000):
        self.cash = initial_cash
        self.daily_pnl = 0.0
        
        # 銘柄リストから動的に辞書を生成（ハードコーディング排除）
        self.positions = {symbol: 0 for symbol in symbols}
        self.entry_prices = {symbol: 0.0 for symbol in symbols}
        self.stop_losses = {symbol: 0.0 for symbol in symbols}
        self.atr_multipliers = {symbol: 2.0 for symbol in symbols} # 後でモデルから上書き
        
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        if not os.path.exists(LOG_PATH):
            df_empty = pd.DataFrame(columns=['timestamp', 'symbol', 'action', 'price', 'quantity', 'realized_pnl', 'cash_balance'])
            df_empty.to_csv(LOG_PATH, index=False)
            
    def log_transaction(self, symbol, action, price, quantity, pnl=0):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df_log = pd.DataFrame([[now_str, symbol, action, price, quantity, pnl, self.cash]], 
                              columns=['timestamp', 'symbol', 'action', 'price', 'quantity', 'realized_pnl', 'cash_balance'])
        df_log.to_csv(LOG_PATH, mode='a', header=False, index=False)
        
        # LINE通知
        if action == "BUY":
            send_line_notification(f"📈 【{symbol} 新規買い】\n単価: {price:,.1f}円 | {quantity}株")
        elif action in ["SELL", "CLOSE_MARKET"]:
            self.daily_pnl += pnl
            status = "利確" if pnl > 0 else "損切"
            send_line_notification(f"📉 【{symbol} 決済売り ({status})】\n単価: {price:,.1f}円\n損益: {pnl:+,.0f}円")
            
    def print_status(self, current_prices):
        total_stock_value = sum(self.positions[sym] * current_prices.get(sym, 0) for sym in self.positions)
        total_value = self.cash + total_stock_value
        profit = total_value - 1000000
        print(f"💰 [ポートフォリオ] 現金: {self.cash:,.0f}円 | 株式: {total_stock_value:,.0f}円 | 総資産: {total_value:,.0f}円 (損益: {profit:+,.0f}円)")

    def send_daily_report(self):
        send_line_notification(f"🏁 【本日トレード終了】\n本日確定損益: {self.daily_pnl:+,.0f}円\n最終現金残高: {self.cash:,.0f}円")

def is_market_open():
    now = datetime.now()
    if now.weekday() > 4: return False
    current_time = now.time()
    morning = datetime.strptime("09:00", "%H:%M").time() <= current_time <= datetime.strptime("11:30", "%H:%M").time()
    afternoon = datetime.strptime("12:30", "%H:%M").time() <= current_time <= datetime.strptime("15:00", "%H:%M").time()
    return morning or afternoon

def run_bot():
    print("=== AutoDayTrade Multi-Symbol Bot 起動 ===")
    symbols = Config.SYMBOLS
    trader = PaperTrader(symbols, initial_cash=1000000)
    fetcher = KabuDataFetcher()
    
    # 銘柄と業界の対応マップ（data_importerのSECTORSと同じ）
    symbol_to_sector = {
        "8306": "banking", "8316": "banking", "8411": "banking", "7182": "banking", "8354": "banking",
        "7203": "auto", "7267": "auto", "7201": "auto", "7270": "auto", "7261": "auto",
        "8035": "semi", "6857": "semi", "6146": "semi", "6723": "semi", "7735": "semi",
        "9432": "telecom", "9433": "telecom", "9434": "telecom", "4443": "telecom", "3994": "telecom",
        "9101": "shipping", "9104": "shipping", "9107": "shipping", "9110": "shipping", "9119": "shipping"
    }

    models = {}
    req_features = {}
    for symbol in symbols:
        sector = symbol_to_sector.get(symbol)
        model_path = os.path.join(Config.MODEL_PATH, f"sector_{sector}.joblib")
        
        if not os.path.exists(model_path):
            print(f"エラー: {symbol} (業界:{sector}) のモデルが見つかりません。")
            continue
            
        saved_data = joblib.load(model_path)
        models[symbol] = saved_data['model']
        req_features[symbol] = saved_data['features']
        trader.atr_multipliers[symbol] = saved_data.get('atr_multiplier', 2.0)
        print(f"[{symbol}] 業界汎用モデル({sector})ロード完了")

    current_prices = {sym: 0.0 for sym in symbols}

    try:
        while True:
            now = datetime.now()
            now_str = now.strftime("%H:%M:%S")
            closing_time = datetime.strptime("15:00", "%H:%M").time()

            # --- 15:00 全銘柄強制終了ロジック ---
            if now.time() >= closing_time:
                print(f"\n[{now_str}] 15:00 大引け。全ポジションを清算します。")
                for symbol in symbols:
                    if trader.positions[symbol] > 0:
                        last_price = current_prices[symbol] if current_prices[symbol] > 0 else trader.entry_prices[symbol]
                        pnl = (last_price - trader.entry_prices[symbol]) * trader.positions[symbol]
                        trader.cash += (trader.positions[symbol] * last_price)
                        trader.log_transaction(symbol, "CLOSE_MARKET", last_price, trader.positions[symbol], pnl=pnl)
                        trader.positions[symbol] = 0
                trader.send_daily_report()
                break

            if not is_market_open():
                print(f"[{now_str}] 市場時間外。待機中...")
                time.sleep(60); continue

            # --- 全銘柄の巡回監視 ---
            for symbol in symbols:
                df_live = fetcher.fetch_historical_data(symbol, Config.EXCHANGE)
                if df_live is None or len(df_live) < 50:
                    time.sleep(1); continue # API制限回避

                df_live = add_features(df_live)
                latest_row = df_live.iloc[-1:]
                current_price = latest_row['Close'].values[0]
                
                # 【修正箇所】ATRを相対値(ATR_Pct)から絶対値(円)に変換して取得
                current_atr = latest_row['ATR_Pct'].values[0] * current_price
                
                current_prices[symbol] = current_price # 状況出力用に保存
                
                # 特徴量の名前が自動的にモデル保存時のもの(req_features)と一致するので
                # predict部分はそのまま動きます
                pred_prob = models[symbol].predict(latest_row[req_features[symbol]])[0]
                
                # 売買判定
                if trader.positions[symbol] == 0:
                    if pred_prob > 0.5:
                        # 全資金を投じるのではなく、1銘柄あたり最大で資金の何割か、という制限を設けても良い（ここでは全額を単元株計算）
                        buy_qty = int(trader.cash // (current_price * 100)) * 100
                        if buy_qty > 0:
                            trader.cash -= buy_qty * current_price
                            trader.positions[symbol], trader.entry_prices[symbol] = buy_qty, current_price
                            trader.stop_losses[symbol] = current_price - (current_atr * trader.atr_multipliers[symbol])
                            trader.log_transaction(symbol, "BUY", current_price, buy_qty)
                            print(f"📈 [{symbol}] 仮想買い {buy_qty}株 @ {current_price}円")
                else:
                    # トレイリングストップ監視
                    trader.stop_losses[symbol] = max(trader.stop_losses[symbol], current_price - (current_atr * trader.atr_multipliers[symbol]))
                    if current_price <= trader.stop_losses[symbol]:
                        pnl = (current_price - trader.entry_prices[symbol]) * trader.positions[symbol]
                        trader.cash += (trader.positions[symbol] * current_price)
                        trader.log_transaction(symbol, "SELL", current_price, trader.positions[symbol], pnl=pnl)
                        print(f"📉 [{symbol}] 仮想売り {trader.positions[symbol]}株 @ {current_price}円 (損益: {pnl:+,.0f}円)")
                        trader.positions[symbol] = 0
                
                time.sleep(1.5) # API制限回避のインターバル

            print(f"\n--- [{now_str}] 監視サイクル完了 ---")
            trader.print_status(current_prices)
            time.sleep(60) # 1分待って次の足へ
            
    except KeyboardInterrupt:
        print("\n=== Bot手動停止 ===")
    finally:
        trader.print_status(current_prices)

if __name__ == "__main__":
    run_bot()