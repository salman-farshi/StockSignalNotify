import requests


def send_telegram(cfg, msg):
    if not cfg.get("enabled"): return
    try:
        url = f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage"
        resp = requests.post(url, data={"chat_id": cfg["chat_id"], "text": msg})
        # print("Telegram response:", resp.status_code, resp.text)
    except Exception as e:
        print("Telegram send error:", e)
    print("📩 Telegram:", msg)


def notify(msg, config):
    if config["telegram"]["enabled"]:
        send_telegram(config["telegram"], msg)
    else:
        print(msg)
