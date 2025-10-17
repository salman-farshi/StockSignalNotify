import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from joblib import load

def train_model(X_seq, y_seq, model_path):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    model.fit(X_seq, y_seq, epochs=20, batch_size=32, verbose=0, callbacks=[es])
    model.save(model_path)
    print(f"✅ Model saved: {model_path}")
    return model

def load_or_train(symbol, df, lookback, data_prep_func):
    model_path = f"{symbol}_model.h5"
    scaler_path = f"{symbol}_scaler.joblib"
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return load_model(model_path)
    X_seq, y_seq = data_prep_func(df, lookback, scaler_path)
    return train_model(X_seq, y_seq, model_path)

def predict_signal(symbol, model, df, lookback):
    scaler = load(f"{symbol}_scaler.joblib")
    features = ["return", "rsi", "macd", "macd_signal"]
    X = df[features].values
    X_scaled = scaler.transform(X)
    if len(X_scaled) < lookback:
        return 0.5
    seq = np.expand_dims(X_scaled[-lookback:], axis=0)
    return float(model.predict(seq, verbose=0)[0][0])
