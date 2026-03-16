@echo off
cd /d %~dp0

echo === [1/2] Trading Bot を起動します (09:00 - 15:00) ===
call .venv\Scripts\activate
python src/bot.py

echo.
echo === [2/2] 本日のデータを用いた再学習を開始します (Optuna) ===
python src/main.py

echo === 全工程が完了しました。最新モデルが生成されました。 ===
pause