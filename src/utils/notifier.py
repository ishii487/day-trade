import requests
import os
from dotenv import load_dotenv

load_dotenv()

def send_line_notification(message):
    """LINE Notifyを使用してメッセージを送信する"""
    token = os.getenv("LINE_NOTIFY_TOKEN")
    if not token:
        print("警告: LINE_NOTIFY_TOKEN が設定されていません。")
        return

    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {token}"}
    data = {"message": f"\n{message}"}
    
    try:
        response = requests.post(url, headers=headers, data=data)
        if response.status_code != 200:
            print(f"LINE通知失敗: {response.status_code} {response.text}")
    except Exception as e:
        print(f"LINE通知エラー: {e}")