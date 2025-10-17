import json, os, time, csv
from datetime import datetime
from utils.notifier import notify
from utils.data_utils import fetch_data, prepare_data
from utils.model_utils import load_or_train, predict_signal, train_model
from utils.scheduler import should_retrain

CONFIG_PATH = "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)


def log_signal(log_file, symbol, prob, signal, price):
    exists = os.path.isfile(log_file)
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp", "symbol", "price", "prob_up", "signal"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), symbol, price, f"{prob:.2f}", signal])


def main():
    print("🚀 AI Trading Bot Started")
    models, last_signals = {}, {}
    buy_signals, sell_signals, hold_signals = {}, {}, {}

    for sym in config["symbols"]:
        df = fetch_data(sym, config["period"], config["interval"])
        models[sym] = load_or_train(sym, df, config["lookback"], prepare_data)
        last_signals[sym] = None
        # buy_signals[sym] = None
        # sell_signals[sym] = None
        # hold_signals[sym] = None

    while True:
        if should_retrain(config):
            print("🧠 Retraining models...")
            for sym in config["symbols"]:
                df = fetch_data(sym, config["period"], config["interval"])
                X_seq, y_seq = prepare_data(df, config["lookback"], f"{sym}_scaler.joblib")
                train_model(X_seq, y_seq, f"{sym}_model.h5")

        for sym in config["symbols"]:
            df = fetch_data(sym, config["period"], config["interval"])
            prob = predict_signal(sym, models[sym], df, config["lookback"])
            price = df["Close"].iloc[-1, 0]  # last row, first column → scalar float
            if prob > 0.6:
                signal = "BUY"
                msg = f"{sym}: {price:.2f} USD | Prob={prob:.2f} → {signal} : {datetime.now():%Y-%m-%d %H:%M}"
                buy_signals[sym] = msg
            elif prob > 0.4:
                signal = "HOLD"
                msg = f"{sym}: {price:.2f} USD | Prob={prob:.2f} → {signal} : {datetime.now():%Y-%m-%d %H:%M}"
                hold_signals[sym] = msg
            else:
                signal = "SELL"
                msg = f"{sym}: {price:.2f} USD | Prob={prob:.2f} → {signal} : {datetime.now():%Y-%m-%d %H:%M}"
                sell_signals[sym] = msg
            # print("BUY:", buy_signals[sym], "\nHOLD:", hold_signals[sym], "\nSELL:", sell_signals[sym])
            log_signal(config["log_file"], sym, prob, signal, price)

        for sym in hold_signals:
            if "HOLD" != last_signals[sym]:
                notify(hold_signals[sym], config)
                last_signals[sym] = "HOLD"
            else:
                print(hold_signals[sym])

        for sym in buy_signals:
            if "BUY" != last_signals[sym]:
               notify(buy_signals[sym], config)
               last_signals[sym] = "BUY"
            else:
                print(buy_signals[sym])

        for sym in sell_signals:
            if "SELL" != last_signals[sym]:
                notify(sell_signals[sym], config)
                last_signals[sym] = "SELL"
            else:
                print(sell_signals[sym])

        print(f"⏳ Sleeping {config['sleep_seconds'] / 60:.0f} min...\n")
        time.sleep(config["sleep_seconds"])

if __name__ == "__main__":
    main()
