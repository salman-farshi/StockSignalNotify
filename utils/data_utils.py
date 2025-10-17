import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import dump

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(period).mean()
    avg_loss = down.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def fetch_data(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    df.dropna(inplace=True)
    df["return"] = df["Close"].pct_change()
    df["rsi"] = compute_rsi(df["Close"])
    df["ema_fast"] = df["Close"].ewm(span=12).mean()
    df["ema_slow"] = df["Close"].ewm(span=26).mean()
    df["macd"] = df["ema_fast"] - df["ema_slow"]
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df.dropna(inplace=True)
    return df

def create_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:(i + lookback)])
        ys.append(y[i + lookback])
    return np.array(Xs), np.array(ys)

def prepare_data(df, lookback, scaler_path):
    features = ["return", "rsi", "macd", "macd_signal"]
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)
    X = df[features].values
    y = df["target"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    dump(scaler, scaler_path)
    X_seq, y_seq = create_sequences(X_scaled, y, lookback)
    return X_seq, y_seq
